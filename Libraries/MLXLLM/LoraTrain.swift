// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXOptimizers

/// Maximum sequence length, in tokens, that ``LoRABatchIterator`` will batch
/// without warning that the data should be pre-split to save memory.
private let maxSequenceLength = 2048

/// Equivalent to `lora.py/iterate_batches()`. Used internally by ``LORATrain``.
struct LoRABatchIterator: Sequence, IteratorProtocol {

    let dataset: [String]
    let batchSize: Int
    let tokenizer: Tokenizer

    let train: Bool

    var indices: [Int]
    var index = 0

    init(dataset: [String], tokenizer: Tokenizer, batchSize: Int, train: Bool) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.train = train

        self.indices = Array(0 ..< dataset.count)
        if train {
            indices.shuffle()
        }
    }

    /// Produce the next batch of tokenized, padded sequences.
    ///
    /// - Returns: a tuple of (inputs batch, targets batch, sequence lengths) where the
    ///   targets are the inputs shifted by one token and the lengths are the unpadded
    ///   token count of each sequence, or `nil` when a non-training iterator has
    ///   consumed the entire dataset. Training iterators reshuffle and iterate
    ///   indefinitely, never returning `nil`.
    mutating func next() -> (MLXArray, MLXArray, MLXArray)? {
        if index >= indices.count {
            if !train {
                return nil
            }

            indices.shuffle()
            index = 0
        }

        let endIndex = Swift.min(index + batchSize, indices.count)

        let batch = (index ..< endIndex)
            .map { tokenizer.encode(text: dataset[indices[$0]]) }
        let lengths = batch.map { $0.count }
        let maxLength = lengths.max() ?? 0

        if maxLength > maxSequenceLength {
            print(
                """
                [WARNING] Some sequences are longer than \(maxSequenceLength) tokens.
                Consider pre-splitting your data to save memory.
                """)
        }

        // pad to the max length
        let batchArray = MLXArray.zeros([lengths.count, maxLength], type: Int32.self)
        for (j, (b, l)) in zip(batch, lengths).enumerated() {
            batchArray[j, 0 ..< l] = MLXArray(b)
        }

        index = endIndex

        return (batchArray[0..., .stride(to: -1)], batchArray[0..., 1...], MLXArray(lengths))
    }
}

/// Collection of functions for adding LoRA adapters to an LLM model, training, fusing and saving/loading weights.
///
/// The typical flow for training is:
///
/// ```swift
/// // load the base model and tokenizer
/// let (model, tokenizer) = try await LLM.load(configuration: ModelConfiguration.mistral7b4bit)
///
/// // add LoRALinear adapter layers
/// LORATrain.convert(model: model, layers: Array(model.loraLinearLayers().suffix(4)))
///
/// // optionally load LoRA weights
/// try LORATrain.loadLoRAWeights(model: model, url: ...)
///
/// // load the train/validation data
/// let train = try loadLoRAData(directory: data, name: "train")
/// let valid = try loadLoRAData(directory: data, name: "valid")
///
/// // train
/// let optimizer = Adam(learningRate: 1e-5)
/// try await LORATrain.train(
///     model: model, train: train, validate: valid, optimizer: optimizer, tokenizer: tokenizer,
///     parameters: LORATrain.Parameters()
/// ) { progress in
///     print(progress)
///     return .more
/// }
/// ```
///
/// At this point the model will be trained and you could do one of the following:
///
/// - ``saveLoRAWeights(model:to:)``: write the LoRA weights to a file
/// - fuse the LoRA weights and convert back into the original model
///     architecture using the LoRA adapter APIs in `MLXLMCommon`
/// - ``evaluate(model:dataset:loss:tokenizer:batchSize:batchCount:)``-- compute the test loss
///     againts a test dataset
/// - use the in memory model as a normal `LLMModel` and evaluate a prompt
///
public enum LORATrain {

    /// Type of a loss function used in LoRA training.
    ///
    /// Given the model, a batch of input tokens, the matching target tokens and the
    /// unpadded length of each sequence, it returns the scalar loss and the number
    /// of tokens that contributed to that loss. See ``loss(model:inputs:targets:lengths:)``
    /// for the default implementation.
    public typealias LoRALossFunction = (Module, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )

    /// LoRA training parameters.
    public struct Parameters: Sendable {
        /// number of prompts to evaluate per iteration.
        public var batchSize = 4

        /// number of iterations to train for.
        public var iterations = 1000

        /// number of training steps between loss reporting.
        public var stepsPerReport = 10

        /// number of steps between validations.
        public var stepsPerEval = 100

        /// number of validations batches, `0` uses the entire validation set.
        public var validationBatches = 10

        /// save the model every N iterations.
        public var saveEvery = 100

        /// save path for the adapter `.safetensors`.
        public var adapterURL: URL?

        /// Create LoRA training parameters.
        ///
        /// - Parameters:
        ///   - batchSize: number of prompts to evaluate per iteration
        ///   - iterations: number of iterations to train for
        ///   - stepsPerReport: number of training steps between loss reporting
        ///   - stepsPerEval: number of steps between validations
        ///   - validationBatches: number of validation batches, `0` uses the entire validation set
        ///   - saveEvery: save the model every N iterations
        ///   - adapterURL: save path for the adapter `.safetensors`
        public init(
            batchSize: Int = 4, iterations: Int = 1000, stepsPerReport: Int = 10,
            stepsPerEval: Int = 100, validationBatches: Int = 10, saveEvery: Int = 100,
            adapterURL: URL? = nil
        ) {
            self.batchSize = batchSize
            self.iterations = iterations
            self.stepsPerReport = stepsPerReport
            self.stepsPerEval = stepsPerEval
            self.validationBatches = validationBatches
            self.saveEvery = saveEvery
            self.adapterURL = adapterURL
        }
    }

    /// Default loss function for LoRA training -- see ``LoRALossFunction``.
    ///
    /// Runs the model on `inputs` and computes the cross entropy of the logits against
    /// `targets`, masking out any padding beyond each sequence's length.
    ///
    /// - Parameters:
    ///   - model: the model to evaluate -- must conform to `LLMModel`
    ///   - inputs: batch of input token ids
    ///   - targets: batch of target token ids
    ///   - lengths: unpadded length of each sequence in the batch
    /// - Returns: the scalar loss and the number of tokens that contributed to it
    public static func loss(model: Module, inputs: MLXArray, targets: MLXArray, lengths: MLXArray)
        -> (
            MLXArray, MLXArray
        )
    {
        // def loss(model, inputs, targets, lengths):

        // run model on inputs -- this function cannot throw (it is used inside the
        // non-throwing `valueAndGrad` closure in `train`), so a model that is not
        // an LLMModel is a programmer error
        guard let model = model as? any LLMModel else {
            fatalError(
                "LORATrain.loss requires a model conforming to LLMModel, got \(type(of: model))")
        }
        let logits = model(inputs, cache: nil).asType(.float32)

        // mask padding tokens
        let lengthMask = MLXArray(0 ..< inputs.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]

        // calculate the loss
        let ntoks = lengthMask.sum()
        let ce = (crossEntropy(logits: logits, targets: targets) * lengthMask).sum() / ntoks
        return (ce, ntoks)
    }

    /// Evaluate the model and dataset and return the loss over the entire dataset.
    ///
    /// - Parameters:
    ///   - model: the model to evaluate
    ///   - dataset: the dataset
    ///   - loss: loss function
    ///   - tokenizer: tokenizer
    ///   - batchSize: number of items from the dataset to evaluate at once
    ///   - batchCount: number of batch elements to evaluate, 0 for all
    /// - Returns: the loss over the enumerate data
    ///
    /// ### See Also
    /// - ``loadLoRAData(directory:name:)``
    public static func evaluate(
        model: Module, dataset: [String], loss: LoRALossFunction = loss, tokenizer: Tokenizer,
        batchSize: Int, batchCount: Int
    ) -> Float {
        var allLosses: [Float] = []
        var tokenCount = 0

        for (iteration, (inputs, targets, lengths)) in LoRABatchIterator(
            dataset: dataset, tokenizer: tokenizer, batchSize: batchSize, train: false
        ).enumerated() {
            let (losses, tokens) = loss(model, inputs, targets, lengths)
            allLosses.append((losses * tokens).item(Float.self))
            tokenCount += tokens.item(Int.self)

            if batchCount != 0 && iteration + 1 >= batchCount {
                break
            }
        }

        return (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
    }

    /// Given a model with LoRA adaptors applied, write adapter weights to a `.safetensors` file.
    ///
    /// - Parameters:
    ///   - model: the model whose trainable (adapter) parameters will be written
    ///   - url: destination of the `.safetensors` file
    /// - Throws: When writing the weights to the `.safetensors` file fails.
    ///
    /// ### See Also
    /// - ``evaluate(model:dataset:loss:tokenizer:batchSize:batchCount:)``
    /// - ``train(model:train:validate:optimizer:loss:tokenizer:parameters:progress:)``
    public static func saveLoRAWeights(model: Module, to url: URL) throws {
        let parameters = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened())
        try save(arrays: parameters, url: url)
    }

    /// Progress event reported to the training callback.
    ///
    /// Passed to the callback given to
    /// ``train(model:train:validate:optimizer:loss:tokenizer:parameters:progress:)``,
    /// covering training loss updates, validation loss updates and adapter weight saves.
    public enum Progress: CustomStringConvertible, Sendable {
        /// a training loss report with throughput statistics.
        case train(
            iteration: Int, trainingLoss: Float, iterationsPerSecond: Double,
            tokensPerSecond: Double)
        /// a validation loss report with the time the validation pass took.
        case validation(iteration: Int, validationLoss: Float, validationTime: Double)
        /// adapter weights were saved to the given url.
        case save(iteration: Int, url: URL)

        /// human readable description of the progress event.
        public var description: String {
            switch self {
            case .train(
                let iteration, let trainingLoss, let iterationsPerSecond, let tokensPerSecond):
                "Iteration \(iteration + 1): training loss \(trainingLoss.formatted()), "
                    + "iterations/sec \(iterationsPerSecond.formatted()), "
                    + "Tokens/sec \(tokensPerSecond.formatted())"
            case .validation(let iteration, let validationLoss, let validationTime):
                "Iteration \(iteration + 1): "
                    + "validation loss \(validationLoss.formatted()), "
                    + "validation time \(validationTime.formatted())s"
            case .save(let iteration, let url):
                "Iteration \(iteration + 1): saved weights to \(url.path())"
            }
        }
    }

    /// Value returned from the ``Progress`` callback indicating whether training should continue or stop.
    public enum ProgressDisposition: Sendable {
        /// stop training.
        case stop
        /// continue training.
        case more
    }

    /// Train (or continue training) LoRA weights.
    ///
    /// - Parameters:
    ///   - model: model to train
    ///   - train: training dataset
    ///   - validate: validate dataset
    ///   - optimizer: optimizer used in training
    ///   - loss: loss function
    ///   - tokenizer: tokenizer
    ///   - parameters: training parameters
    ///   - progress: progress callback
    /// - Throws: When saving the adapter weights fails -- see ``saveLoRAWeights(model:to:)``.
    public static func train(
        model: Module, train: [String], validate: [String], optimizer: Optimizer,
        loss: @escaping LoRALossFunction = loss, tokenizer: Tokenizer, parameters: Parameters,
        progress: (Progress) -> ProgressDisposition
    ) throws {
        // def train(model, train_set, val_set, optimizer, loss, tokenizer, args)

        /// Report `event` to the caller's `progress` callback and return `true`
        /// when the callback asks training to stop.
        func checkProgress(_ event: Progress) -> Bool {
            progress(event) == .stop
        }

        let lossValueGrad = valueAndGrad(model: model) { model, arrays in
            let (ce, ntoks) = loss(model, arrays[0], arrays[1], arrays[2])
            return [ce, ntoks]
        }

        var losses: [Float] = []
        var tokenCount = 0

        var start = Date.timeIntervalSinceReferenceDate

        /// Report the mean training loss and throughput every `stepsPerReport`
        /// iterations, resetting the accumulated statistics afterwards.
        ///
        /// - Returns: `true` when the progress callback asks training to stop.
        func reportTrainingProgress(iteration: Int) -> Bool {
            guard (iteration + 1) % parameters.stepsPerReport == 0 else {
                return false
            }

            let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
            let now = Date.timeIntervalSinceReferenceDate

            let iterationsPerSecond = Double(parameters.stepsPerReport) / (now - start)
            let tokensPerSecond = Double(tokenCount) / (now - start)

            if checkProgress(
                .train(
                    iteration: iteration, trainingLoss: trainingLoss,
                    iterationsPerSecond: iterationsPerSecond, tokensPerSecond: tokensPerSecond))
            {
                return true
            }

            losses.removeAll()
            tokenCount = 0
            start = Date.timeIntervalSinceReferenceDate
            return false
        }

        /// Compute and report the validation loss on the first iteration and every
        /// `stepsPerEval` iterations thereafter.
        ///
        /// - Returns: `true` when the progress callback asks training to stop.
        func reportValidationLoss(iteration: Int) -> Bool {
            guard iteration == 0 || (iteration + 1) % parameters.stepsPerEval == 0 else {
                return false
            }

            let validationStart = Date.timeIntervalSinceReferenceDate
            let validationLoss = evaluate(
                model: model, dataset: validate, loss: loss, tokenizer: tokenizer,
                batchSize: parameters.batchSize, batchCount: parameters.validationBatches)
            let now = Date.timeIntervalSinceReferenceDate

            if checkProgress(
                .validation(
                    iteration: iteration, validationLoss: validationLoss,
                    validationTime: now - validationStart))
            {
                return true
            }

            start = Date.timeIntervalSinceReferenceDate
            return false
        }

        /// Save the adapter weights every `saveEvery` iterations when
        /// ``Parameters/adapterURL`` is configured.
        ///
        /// - Returns: `true` when the progress callback asks training to stop.
        /// - Throws: When saving the adapter weights fails.
        func saveAdapterIfNeeded(iteration: Int) throws -> Bool {
            guard let adapterURL = parameters.adapterURL,
                (iteration + 1) % parameters.saveEvery == 0
            else {
                return false
            }

            try saveLoRAWeights(model: model, to: adapterURL)

            if checkProgress(.save(iteration: iteration, url: adapterURL)) {
                return true
            }

            start = Date.timeIntervalSinceReferenceDate
            return false
        }

        for (iteration, (inputs, targets, lengths)) in LoRABatchIterator(
            dataset: train, tokenizer: tokenizer, batchSize: parameters.batchSize, train: true
        ).enumerated() {
            // forward and backward pass
            let (resultArray, grad) = lossValueGrad(model, [inputs, targets, lengths])
            let lvalue = resultArray[0]
            let tokens = resultArray[1]

            // model update
            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer, lvalue)

            // record loss
            losses.append(lvalue.item(Float.self))
            tokenCount += tokens.item(Int.self)

            if reportTrainingProgress(iteration: iteration) {
                break
            }

            if reportValidationLoss(iteration: iteration) {
                break
            }

            if try saveAdapterIfNeeded(iteration: iteration) {
                break
            }

            if iteration + 1 >= parameters.iterations {
                break
            }
        }
    }
}

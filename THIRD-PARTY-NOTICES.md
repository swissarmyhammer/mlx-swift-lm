# Third-Party Notices

The `mlx-swift-lm` repository contains code from the projects in this list. Each
project has its license below. The license of this repository is in the
`LICENSE` file.

For the rules that tell you how to attribute ported code, read the section
"Attribution for Ported Code" in `CONTRIBUTING.md`.

## Projects

- `osaurus-ai/vmlx-swift-lm` — MIT license. Each ported DeepSeek-V4 file in this
  repository comes from this project.
- `scouzi1966/mlx-swift-lm` — MIT license. This is a Swift reference for
  DeepSeek-V4 model code.
- `scouzi1966/maclocal-api` — MIT license. This is an integration reference.

No project in this list is a GitHub fork of this repository. There is no git
ancestry between them and this repository. A person transcribed each port by
hand and added the attribution by hand.

---

## osaurus-ai/vmlx-swift-lm

Source: https://github.com/osaurus-ai/vmlx-swift-lm

```
MIT License

Copyright (c) 2024 ml-explore
Copyright (c) 2026 Osaurus contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Two different names, and both are correct

`Osaurus AI` and `Osaurus contributors` are two different names. Each name comes
from a different file:

- This notice keeps `Osaurus contributors`, because that is the name in the
  LICENSE above.
- The header block in `CONTRIBUTING.md` keeps `Osaurus AI`, because that is the
  name in the header of the source file that a person read.

Each name is a true transcription of the file it comes from. Do not change one
name into the other.

This notice does not tell you about the headers of the other files in that
project. Before you port a file, read the header of that file and find the
commit SHA. Write those two items in the header block of the new file.

---

## scouzi1966/mlx-swift-lm

Source: https://github.com/scouzi1966/mlx-swift-lm

```
MIT License

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

A person read `Libraries/MLXLLM/Models/DeepseekV4.swift` in that repository, and
found this header at the top of it:

```
// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
```

That header shows that the code comes from `osaurus-ai/vmlx-swift-lm`. Thus the
ported DeepSeek-V4 files in this repository give the copyright to Osaurus AI.

---

## scouzi1966/maclocal-api

Source: https://github.com/scouzi1966/maclocal-api

```
MIT License

Copyright (c) 2025 MacLocalAPI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

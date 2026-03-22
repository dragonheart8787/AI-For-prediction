---
license: apache-2.0
tags:
- custom
- pytorch
---

# demo_model

這是一個演示模型

## 使用方式

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("demo_export")
tokenizer = AutoTokenizer.from_pretrained("demo_export")
```

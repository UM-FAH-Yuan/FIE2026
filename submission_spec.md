# FIE2026提交文件说明（Submission Format Specification for FIE2026）

参赛者需提交预测结果文件（以下简称 **output 文件**）。假设评测集包含
**5000（待定）
条数据**，提交文件必须满足以下格式与规则，否则系统将返回对应错误信息。

Participants are required to submit a prediction result file (hereinafter referred to as the **output file**). Assuming the evaluation set contains **5,000 (TBD)  instances**, the submission file must comply with the following format and rules; otherwise, the system will return the corresponding error message.

------------------------------------------------------------------------

# 1. 文件要求 File Requirements

提交文件需满足以下要求：
The submission file must meet the following requirements:

-   **文件名File name：** `output.json`
-   **文件格式File format：** JSON
-   **文件编码File encoding：** UTF-8

提交文件应为一个 **JSON 数组（JSON Array）**，数组中包含 **5000（待定）条预测结果**，每条结果为一个 JSON 对象。
The submission file must be a **JSON Array** containing **5,000 (TBD) prediction results**, where each result is a JSON object.

示例结构Example structure：

``` json
[
  {"track": "pr", "id": 1, "factivity": "TRUE", "confidence": 0.87},
  {"track": "pr", "id": 2, "factivity": "FALSE", "confidence": 0.73},
  {"track": "pr", "id": 3, "factivity": "UNCERTAIN", "confidence": 0.50}
]
```
``` json
[
  {"track": "ft", "id": 1, "factivity": "TRUE", "confidence": 0.87},
  {"track": "ft", "id": 2, "factivity": "FALSE", "confidence": 0.73},
  {"track": "ft", "id": 3, "factivity": "UNCERTAIN", "confidence": 0.50}
]
```

------------------------------------------------------------------------

# 2. 文件校验规则 File Validation Rules

## 2.1 文件格式校验 File Format Validation

若提交文件不是合法 JSON 文件，系统将返回错误：
If the submission file is not a valid JSON file, the system will return the following error:

    file format error: JSON is expected.

## 2.2 文件编码校验 File Encoding Validation

若提交文件编码不是 UTF-8，系统将返回错误：
If the submission file is not encoded in UTF-8, the system will return the following error:

    file encoding error: UTF-8 is expected.

------------------------------------------------------------------------

# 3. 数据条数要求 Data Count Requirements

提交文件中的数据条数必须 **严格等于正式评测集中的数据条数**。
The number of data in the submission file must be **strictly equal to the number of data in the evaluation set. **.

否则，系统将返回错误：
If not, the system will return the following error:

    object quantity error: 5000 objects are expected.

------------------------------------------------------------------------

# 4. 数据字段要求 Data Field Requirements

提交文件中的每条数据 **必须且只能包含以下四个字段**：
Each data instance in the submission file **must contain exactly the following 4 fields**:

  |字段名 Field Name      |数据类型 Data Type           |说明Description
  |------------ |-------------- |------------
  |track       |string         |赛道名称 Name of track
  |id           |integer        |数据编号 Data identifier 
  |factivity    |string         |事实性判断 Factivity judgment 
  |confidence   |float          |模型置信度 Model confidence score 

## 4.1 字段数量检查 Field Count Check

若某条数据字段数量不为 4，系统将返回错误：
If a data instance does not contain exactly 4 fields, the system will return the following error:

    field quantity error : 4 fields are expected.

其中 `XXX` 为 output 文件中的行号。
where `XXX` is the line number in the output file.

## 4.2 字段名称检查 Field Name Check

若字段名称不符合要求（必须为`track`、`id`、`factivity`、`confidence`），系统将返回错误：
If the field names do not meet the requirements (must be `track`, `id`, `factivity`, and `confidence`), the system will return the following error:

    field name error : "track", "id", "factivity", "confidence" are expected.

------------------------------------------------------------------------

# 5. 字段取值规则 Field Value Rules

## 5.1 track

`track`字段取值只能为`"pr"`或`"ft"`

队伍在提交文件是必须检查所提交的文件与赛道是否匹配。如不匹配，则会返回。

field value error : "track" must be "pr".（提示词赛道）
field value error: "track" must be "ft".（微调赛道）

## 5.2 id 

`id` 字段必须满足以下要求：
The `id` field must meet the following requirements:

-   取值范围为 **[1, 5000]** 的整数 Must be an integer within the range **[1, 5000]**
-   每个 `id` **只能出现一次** Each `id` value **must appear exactly once**
-   必须与评测集 **一一对应** Must **correspond one-to-one** with the evaluation set
-   **必须按 id 升序排列** **Must be sorted in ascending order**

示例Example：

    1, 2, 3, 4, ..., 5000

若不满足上述条件，系统将返回错误：
If any of the above conditions are not met, the system will return the following error:

    field value error : "id" must be an integer in [1, 5000] and sorted in ascending order.

## 5.3 factivity 

`factivity` 字段的合法取值为：
The valid values for the `factivity` field are:

    TRUE
    FALSE
    UNCERTAIN

若出现其他取值，系统将返回错误：
If any other value appears, the system will return the following error:

    field value error : "factivity" must be one of {"TRUE", "FALSE", "UNCERTAIN"}.

## 5.4 confidence 

### 情况一 Case 1："factivity" = "TRUE" or "FALSE"

当 `factivity` 为 `TRUE` 或 `FALSE` 时，`confidence` 字段必须满足：
When `factivity` is `TRUE` or `FALSE`, the `confidence` field must satisfy:

-   不允许为空  Must not be empty
-   必须为数值类型  Must be a numeric type
-   取值范围 (0.50, 1.00\]  Value range: (0.50, 1.00]
-   不包含 0.50  0.50 is not included
-   包含 1.00  1.00 is included
-   保留两位小数  Must be rounded to two decimal places

合法示例 Valid examples：

    0.51
    0.76
    0.95
    1.00

若不满足要求，系统将返回错误：
If the requirements are not met, the system will return the following error:

    field value error : "confidence" must be a number in (0.50, 1.00] with two decimal places.

### 情况二 Case 2："factivity" = "UNCERTAIN"

当 `factivity` 为 `UNCERTAIN` 时，`confidence` 字段必须满足：
When `factivity` is `UNCERTAIN`, the `confidence` field must satisfy:

-   不允许为空 Must not be empty
-   必须为数值类型 Must be a numeric type
-   字段值固定为 **0.50** The field value is fixed at **0.50**.

若不满足要求，系统将返回错误：
If the requirement is not met, the system will return the following error:

    field value error : "confidence" must be 0.50 when "factivity" is "uncertain".

------------------------------------------------------------------------

# 6. 提交文件示例 Submission File Example

示例Example（output.json）：

``` json
[
  {"track": "pr", "id": 1, "factivity": "TRUE", "confidence": 0.87},
  {"track": "pr", "id": 2, "factivity": "FALSE", "confidence": 0.73},
  {"track": "pr", "id": 3, "factivity": "UNCERTAIN", "confidence": 0.50},
  {"track": "pr", "id": 4, "factivity": "TRUE", "confidence": 0.91}
]
```

------------------------------------------------------------------------

# 7. 常见错误示例 Common Error Examples

 | 错误类型 Error Type                     |示例 Examples|
  |-------------------------------- |----------------------|
  |数据条数错误 Incorrect data count|4999 条或 5001 条 4,999 or 5,001 instances |
  |字段数量错误 Incorrect field count|少于或多于 4 个字段 Fewer or more than 4 fields|
  |字段名称错误 Incorrect field name |`fact`, `score`|
  |id 未升序 id not in ascending order| `1,3,2`|
  |id 重复 Duplicate id   | 两条记录 `"id" = 15` Two records with `"id" = 15`|
  |confidence 超范围 confidence out of range| `0.40`, `-0.85`|
  |confidence 小数位错误 confidence with incorrect decimal places |`0.7563`|
  |UNCERTAIN 但 confidence 不为0.50  UNCERTAIN but confidence is not 0.50|`{"factivity": "UNCERTAIN", "confidence": 0.35}`|


------------------------------------------------------------------------

# 8. 错误信息说明 Error Message Specification

所有错误信息遵循统一格式：
All error messages follow a unified format:

`<error type> : <expected condition>`

示例Example：

  |Error Message            |含义 Meaning   
  |------------------------ |--------------------
  |`file format error`      |文件不是合法 JSON The file is not a valid JSON file 
  |`file encoding error`    |文件编码不是 UTF-8 The file encoding is not UTF-8 
 | `object quantity error`  |数据条数不为 5000 The number of data instances is not 5,000 
|  `field quantity error`   |字段数量不为 4 The number of fields is not 4
|  `field name error`       |字段名称不正确 The field names are incorrect
|  `field value error`      |字段取值不符合要求 The field values do not meet the requirements


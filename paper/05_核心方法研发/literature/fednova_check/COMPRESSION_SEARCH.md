# FedNova 与压缩：一次有限检索

核查日期：2026-11-04。沿用已读的 paper-lookup skill 与 arXiv API reference。仅执行一次查询，保存原始与解析记录；没有追加查询、取得其他 primary 全文、追引文、访问 repo 或训练。

## 查询与原始记录

- 数据库：arXiv Atom API。
- 查询：`all:FedNova AND (all:compression OR all:compressed OR all:sparsification OR all:quantization)`。
- Endpoint：`https://export.arxiv.org/api/query?search_query=all:FedNova%20AND%20%28all:compression%20OR%20all:compressed%20OR%20all:sparsification%20OR%20all:quantization%29&start=0&max_results=40&sortBy=relevance&sortOrder=descending`。
- 原始返回：`compression_arxiv_raw.xml`。
- 解析返回：`compression_arxiv_parsed.json`。
- API明确返回 `totalResults=0`、`startIndex=0`、`itemsPerPage=40`（请求页容量）；没有条目可供本次读取标题/摘要。

## 结论与证据边界

本次精确查询**未命中直接邻居**，因而不能据本次返回列出“FedNova+压缩”的已核实论文，也没有任何标题或摘要可支持具体压缩机制结论。

零命中只表示该查询在 arXiv 索引中的检索结果为空，不表示该组合不存在，也不能支持“FedNova尚未与压缩结合”的主张。以缩写为必需词会遗漏只在正文/实验中出现FedNova、使用其他命名或未被该检索项覆盖的论文；本次按约束不扩检索。

主agent对原版FedNova的唯一primary全文核查应独立提供归一公式和适用条件；此处没有新增压缩论文证据。因此以下问题在本次范围内均未核实：稀疏化/量化在FedNova归一之前还是之后、误差反馈的状态与作用对象、各client不同支持下如何归一、误差界与具体通信预算。

用户允许合理组合已有机制；本次零命中不构成停止组合研究的理由，也不构成novelty或收益证据。若先做实现，宜称“采用FedNova式归一的压缩组合/变体”，准确交代顺序、状态及对照，不将本次检索写成领域穷尽。

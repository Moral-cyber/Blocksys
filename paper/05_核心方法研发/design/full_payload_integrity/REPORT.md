本地完整维度集成：183/183检查通过，simulation_only；仅真实训练生成首包接入，非真实联邦训练端到端。

冻结fedlc_uniform3首包54067B，真实解码136874维；核对codec源码及spec固定哈希，以原SmallCNN布局构造全零模型。两客户端均重放同包q，更新人为设q＋不同确定扰动，初始EF为零；并未复现历史delta或重新编码。固定权重0.25/0.75聚合与独立预期逐位一致，误差0；因q相同，不区分不同更新上的权重正确性。

REGISTERED/ACCEPTED均不提交EF；INCORPORATED后只提交一次。实际子进程验证SQL提交前崩溃回滚、服务端提交后丢ack、客户端提交后丢ack及重开；pending逐字节重传不调用encoder，3次重放不再解码/累计。32项载荷scale/code/元数据/tag及base/layout/codec版本、spec漂移拒绝，模型/EF保持原值。

每客户端应用封装=54373B wire＋32B HMAC-SM3=54405B；其中原压缩包54067B。3个SQLite文件实测合计14045184B，完整性均ok；逐库大小见JSON。10个受保护输入前后哈希一致。随机测试密钥600权限、独立private_test_keys目录，结果树无key。

边界沿用v2：信任持钥服务器、数据库、查询/Receipt及真实decoder；spec/source相同不证明执行，HMAC不证明更新诚实。没有真实链、网络或分布式原子性；无数据/TEST读取、无新训练。LOCAL_CHECK_ONLY / REVIEW_UNAVAILABLE。

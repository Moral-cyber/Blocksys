不同真实首包的完整维度集成：186/186检查通过，simulation_only。保留原同包试验。

c1用CE uniform3、c2用FedLC uniform3既有首包，各54067B、解码136874维；两wire与解码哈希不同，77469个坐标不同。固定权重0.25/0.75，以NumPy FP64独立计算0.25q_CE＋0.75q_FedLC；SQLite模型及累计量逐位一致，最大误差0。错误等权预期最大相差0.0153089203。按原布局回填SmallCNN的FP32舍入也独立核对。

初始模型/EF为零；两客户端更新人为设各自q＋不同确定扰动，encoder仅重放对应真实wire，不是历史训练delta或新量化。REGISTERED/ACCEPTED不提交EF，INCORPORATED后残差只提交一次。实际子进程验证SQL提交前崩溃回滚、服务端及客户端提交后丢ack重开；pending逐字节重传不重编码，3次重放不再次聚合或提交EF。

32项scale/code/tag/元数据及base/layout/codec/spec拒绝均通过。每客户端封装54373B wire＋32B HMAC-SM3＝54405B；3个SQLite文件实测合计14045184B，完整性均ok。13个受保护输入前后哈希不变；随机测试key600权限、结果树无key。

这是真实训练首包的合成协议集成，非真实联邦训练端到端。信任持钥服务器、数据库、查询/Receipt及decoder；spec/source身份不证明执行，HMAC不证明更新诚实。无真实链/网络/分布式原子性，无数据/TEST读取、无新训练。LOCAL_CHECK_ONLY / REVIEW_UNAVAILABLE。

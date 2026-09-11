# 论文引用核查记录

核查日期：2026-09-12。范围：67 条参考文献（66 篇论文/报告及 1 条地图许可来源）、正文全部引用位置，以及表 2.1 的 8 篇重点比较文献。

## 核查方法与结论

逐条比对题名、参考文献中列出的作者、年份、卷期/页码和 URL。优先使用会议/期刊官网及出版社提交的 Crossref 元数据；引用用法对照原文摘要与相关内容，重点检查方法归属、比较范围和结论的适用条件。本次并非对全部 66 篇论文逐页精读，也不把 HTTP 200、搜索命中或标题相同视为完整内容核验。

67 个正文引用身份均有对应参考文献，67 条参考文献均被引用；其中 30 篇采用 2024–2026 正式发表年份。未发现题名与 DOI/论文页面指向另一篇论文的情况。访问受限与失效链接分开记录；不宣称每个出版商全文页面均可直接打开。

## 已修改

- 引言：将 Jain/Wallace、Serrano/Smith 和 Liu 的实证质疑，与 Wiegreffe/Pinter 反对全面否定 attention 解释性的立场分别表述，同时注明前者研究背景，避免把 NLP 结果直接写成车队调度结论。
- 第 2.2 节：区分“删除重要证据后的预测损失”和“保留充分证据后的预测保持”，并补上 ERASER 出处。
- 第 2.5 节：将 Adebayo 的标签随机化准确表述为在随机标签上重新训练；不是单纯更改输入标签。
- 第 3.2 节：地图位置与提取记录由项目网络文件支撑；OpenStreetMap 版权链接专用于地图署名和 ODbL 许可。
- 第 3.5–3.6 节：补充 AoI 定义与 ERASER 方法出处，明确 matched-random gains 和 DEF 组合由本研究定义；未改变任何计算公式或数值。
- 作者写法：恢复 Veličković、Díaz-Rodríguez、Öztireli 的正确拼写；GNNExplainer 按当前 NeurIPS 官方记录将首作者缩写写为 Ying, Z.。此项是与所引版本记录对齐，不是认定其为另一位作者。
- 两篇 Lu（2024）分别为 Shengyao Lu 和 Wenhao Lu 领衔的不同团队：正文改用 S. Lu et al. (2024) 与 W. Lu et al. (2024)，参考文献取消 2024a/2024b，避免暗示同一作者组。
- Azzolin（2025）改用直接可读的 ICLR 正式论文集页面；MAPPO 改用 NeurIPS 正式论文集页面。原 OpenReview/DOI 不是被判定为错误，而是改善访问路径。
- 为全部 67 条参考文献添加真正的外部超链接，目标 URL 排除句末标点。

## 重点比较文献的引用用法

| 文献 | 核查结果 |
|---|---|
| Zheng et al. (2024) | 分布偏移影响 fidelity、提出 robust fidelity 的表述有依据；未写成已解决本研究的动作删除问题。 |
| Li et al. (2024) | 预测保持正确时结构扰动仍可大幅改变解释，表述准确；未等同于 stale-feature 干预。 |
| Azzolin et al. (2025) | 指标不可互换及架构依赖有原文依据；未将 DEF 宣称为普遍因果指标。 |
| Shin et al. (2025) | GAtt 是基于 computation tree 的边归因，表述准确；正文明确没有在本研究实现或验证 GAtt。 |
| Liu and Xie (2025) | 连续变化的边权、message-flow/layer-edge 归因及八个数据集与官方摘要一致。 |
| Liu et al. (2025) | curvature/resistance、九个数据集和三类图任务、无需重训目标模型的表述与原文一致。 |
| Azzolin et al. (2026) | 模型可有好预测但退化解释与推断无关，且部分指标识别失败，表述准确。 |
| Saha and Bandyopadhyay (2026) | sufficiency risk 及有限样本不确定性有依据；正文明确没有把其 model-level 保证转移给本研究的 local DEF。 |

其他新近方法的用途也已核对：GOAt、两种不同的 MAGE、RegExplainer、GraphTrail、GNNBoundary、GraphNarrator、COViz、causal state distillation、DFBT、MAGI、MRCNet 及通信策略综合均按其各自任务进行方法比较，而非作为本研究直接跑过的性能基线。BMG-Q、CoopRide 与 DualG-MARL 仅用于调度任务背景。

## 年份与版本特别说明

- Bekkemoen：2023 年 online-first，2024 年正式卷期 113(1)，保留 2024。
- Milani：所引 ACM 正式卷期为 2024 年 56(7)，保留 2024。
- Liu 的 graph-curvature 论文：官网页面生成/出版元数据日期含 2026，但归属 NeurIPS 2025、卷 38，保留会议年份 2025。
- Bouteiller：ICLR 2021 官方会议页面把 Yann Bouteiller 列在首位；arXiv v3 把 Simon Ramstedt 列在首位（共同一作）。本论文引用会议版本，因此保留 Bouteiller et al. (2021)，不混用 arXiv 版本的作者顺序。[会议记录](https://iclr.cc/virtual/2021/poster/3078)
- Yuan 的 TPAMI survey：Crossref 仍返回 early-access 2022、1–19 页记录；IEEE 页面触发访问验证。本次保留原有 2023、45(5)、5782–5799 的正式卷期写法，但该最终卷期未能在本次直接读取的出版社页面中完成复核，建议定稿前人工打开 IEEE citation export 再确认。题名、作者和 DOI 已核对。[IEEE 页面](https://ieeexplore.ieee.org/document/9875989/)

## 逐条书目信息及链接检查

表中“元数据一致”指当前列出的题名、作者及年份在可读取的正式记录中一致；特别版本差异见上文。出版社页面访问限制不影响 DOI 身份核验，但不能据此声称全文可直接访问。

| 序号 | 文献 | 元数据/处理 | 链接检查 |
|---|---|---|---|
| 1 | Abnar, S. and Zuidema, W. (2020) | 元数据一致 | [已读取正式页面](https://doi.org/10.18653/v1/2020.acl-main.385) |
| 2 | Adebayo, J. et al. (2018) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) |
| 3 | Amara, K. et al. (2022) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v198/amara22a.html) |
| 4 | Amitai, Y., Septon, Y. and Amir, O. (2024) | 元数据一致 | [已读取正式页面](https://doi.org/10.1609/aaai.v38i9.28863) |
| 5 | Ancona, M., Ceolini, E., Öztireli, C. and Gross, M. (2018) | 姓名拼写规范；其余一致 | [OpenReview 验证受限；由作者稿/会议记录交叉核对](https://openreview.net/forum?id=Sy21R9JAW) |
| 6 | Armgaan, B., Dalmia, M., Medya, S. and Ranu, S. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/df2d51e1d3e899241c5c4c779c1d509f-Abstract-Conference.html) |
| 7 | Azzolin, S., Longa, A., Teso, S. and Passerini, A. (2025) | 元数据一致 | [已改为可读取的正式论文集页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7969d456aede0ce6b36fb47cc64c495c-Abstract-Conference.html) |
| 8 | Azzolin, S., Teso, S., Lepri, B., Passerini, A. and Malhotra, S. (2026) | 元数据一致 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/75a58cfb4d0de35db7663714e0f52dfa-Abstract-Conference.html) |
| 9 | Bekkemoen, Y. (2024) | 按正式卷期/会议年份；见版本说明 | [已读取正式页面](https://doi.org/10.1007/s10994-023-06479-7) |
| 10 | Bouteiller, Y., Ramstedt, S., Beltrame, G., Pal, C.J. and Binas, J. (2021) | 会议版作者顺序一致；与 arXiv 不同 | [OpenReview 验证受限；由作者稿/会议记录交叉核对](https://openreview.net/forum?id=QFYnKlBJYR) |
| 11 | Bui, N., Nguyen, H.T., Nguyen, V.A. and Ying, R. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v235/bui24b.html) |
| 12 | DeYoung, J. et al. (2020) | 元数据一致 | [已读取正式页面](https://doi.org/10.18653/v1/2020.acl-main.408) |
| 13 | Ding, S., Du, W., Ding, L., Guo, L. and Zhang, J. (2024) | 元数据一致 | [已读取正式页面](https://doi.org/10.1609/aaai.v38i16.29682) |
| 14 | Greydanus, S. et al. (2018) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v80/greydanus18a.html) |
| 15 | He, J., Rafiey, A., Mishne, G. and Wang, Y. (2025) | 元数据一致 | [出版社访问受限；DOI 注册记录匹配](https://doi.org/10.1145/3711896.3736947) |
| 16 | Heuillet, A., Couthouis, F. and Díaz-Rodríguez, N. (2021) | 姓名拼写规范；其余一致 | [已读取正式页面](https://doi.org/10.1016/j.knosys.2020.106685) |
| 17 | Hong, S., Liu, Y., Li, Z., Li, S. and He, Y. (2024) | 元数据一致 | [已读取正式页面](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_Multi-agent_Collaborative_Perception_via_Motion-aware_Robust_Communication_Network_CVPR_2024_paper.html) |
| 18 | Hooker, S. et al. (2019) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html) |
| 19 | Hu, Y., Feng, S. and Li, S. (2025) | 元数据一致 | [DOI 重定向至 IEEE；页面验证受限](https://doi.org/10.1109/TITS.2025.3595653) |
| 20 | Iqbal, S. and Sha, F. (2019) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v97/iqbal19a.html) |
| 21 | Jacovi, A. and Goldberg, Y. (2020) | 元数据一致 | [已读取正式页面](https://doi.org/10.18653/v1/2020.acl-main.386) |
| 22 | Jain, S. and Wallace, B.C. (2019) | 元数据一致 | [已读取正式页面](https://aclanthology.org/N19-1357/) |
| 23 | Kaul, S., Yates, R. and Gruteser, M. (2012) | 元数据一致 | [DOI 重定向至 IEEE；页面验证受限](https://doi.org/10.1109/INFCOM.2012.6195689) |
| 24 | Li, J., Pang, M., Dong, Y., Jia, J. and Wang, B. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v235/li24bd.html) |
| 25 | Lin, K. et al. (2018) | 元数据一致 | [出版社访问受限；DOI 注册记录匹配](https://doi.org/10.1145/3219819.3219993) |
| 26 | Liotet, P., Maran, D., Bisi, L. and Restelli, M. (2022) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v162/liotet22a.html) |
| 27 | Lipton, Z.C. (2018) | 元数据一致 | [出版社访问受限；DOI 注册记录匹配](https://doi.org/10.1145/3236386.3241340) |
| 28 | Liu, Y. and Xie, S. (2025) | 元数据一致 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/de88d78c875d93a2098dc49a0d0422da-Abstract-Conference.html) |
| 29 | Liu, Y. et al. (2022) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v162/liu22i.html) |
| 30 | Liu, Y., Zhang, X., Xie, S. and Xiong, H. (2025) | 按正式卷期/会议年份；见版本说明 | [已读取正式页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2c2e95b75a10adbd2359f8ed5c0a38cd-Abstract-Conference.html) |
| 31 | Lopez, P.A. et al. (2018) | 元数据一致 | [DOI 重定向至 IEEE；页面验证受限](https://doi.org/10.1109/ITSC.2018.8569938) |
| 32 | Lowe, R. et al. (2017) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html) |
| 33 | Lu, S., Mills, K.G., He, J., Liu, B. and Niu, D. (2024) | 用首作者名字缩写区分；年份保留 2024 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/92ee07d8a2c8f5ec08eff83f9eff0c1b-Abstract-Conference.html) |
| 34 | Lu, W., Zhao, X., Fryen, T., Lee, J.H., Li, M., Magg, S. and Wermter, S. (2024) | 用首作者名字缩写区分；年份保留 2024 | [已读取正式页面](https://proceedings.mlr.press/v236/lu24a.html) |
| 35 | Luo, D. et al. (2020) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html) |
| 36 | Madumal, P. et al. (2020) | 元数据一致 | [已读取正式页面](https://doi.org/10.1609/aaai.v34i03.5631) |
| 37 | Milani, S. et al. (2024) | 按正式卷期/会议年份；见版本说明 | [出版社访问受限；DOI 注册记录匹配](https://doi.org/10.1145/3616864) |
| 38 | Mott, A. et al. (2019) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper/2019/hash/e9510081ac30ffa83f10b68cde1cac07-Abstract.html) |
| 39 | OpenStreetMap contributors (n.d.) *Copyright and license*. Available at: https://www.openstreetmap.org/copyright (Accessed: 12 September 2026). | 地图许可来源；无发表年份 | [已读取正式页面](https://www.openstreetmap.org/copyright) |
| 40 | Pan, B. et al. (2025) | 元数据一致 | [已读取正式页面](https://doi.org/10.18653/v1/2025.acl-long.2) |
| 41 | Qin, Z., Zhu, H. and Ye, J. (2022) | 元数据一致 | [已读取正式页面](https://doi.org/10.1016/j.trc.2022.103852) |
| 42 | Rashid, T. et al. (2020) | 元数据一致 | [已读取正式页面](https://jmlr.org/papers/v21/20-081.html) |
| 43 | Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) | 元数据一致 | [出版社访问受限；DOI 注册记录匹配](https://doi.org/10.1145/2939672.2939778) |
| 44 | Saha, S. and Bandyopadhyay, S. (2026) | 元数据一致 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c354d2a64a71776538d6dbbc38285216-Abstract-Conference.html) |
| 45 | Scarselli, F. et al. (2009) | 元数据一致 | [DOI 重定向至 IEEE；页面验证受限](https://doi.org/10.1109/TNN.2008.2005605) |
| 46 | Serrano, S. and Smith, N.A. (2019) | 元数据一致 | [已读取正式页面](https://doi.org/10.18653/v1/P19-1282) |
| 47 | Sha, J. et al. (2026) | 元数据一致 | [已读取正式页面](https://doi.org/10.1038/s41598-026-35004-8) |
| 48 | Shin, Y.-M., Li, S., Cao, X. and Shin, W.-Y. (2025) | 元数据一致 | [已读取正式页面](https://doi.org/10.1609/aaai.v39i19.34254) |
| 49 | Soudijani, S. and Dimitrova, R. (2025) | 元数据一致 | [已读取正式页面](https://doi.org/10.24963/ijcai.2025/30) |
| 50 | Tabassi, E. (2023) *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. NIST AI 100-1. Gaithersburg, MD: National Institute of Standards and Technology. Available at: https://doi.org/10.6028/NIST.AI.100-1. | 元数据一致 | [已读取正式页面](https://doi.org/10.6028/NIST.AI.100-1) |
| 51 | Varbella, A., Amara, K., Gjorgiev, B., El-Assady, M. and Sansavini, G. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c7caf017cbbca1f4b368ffdc7bb8f319-Abstract-Datasets_and_Benchmarks_Track.html) |
| 52 | Vaswani, A. et al. (2017) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |
| 53 | Veličković, P. et al. (2018) | 姓名拼写规范；其余一致 | [OpenReview 验证受限；由作者稿/会议记录交叉核对](https://openreview.net/forum?id=rJXMpikCZ) |
| 54 | Wang, J. et al. (2025) | 元数据一致 | [出版社访问受限；DOI 注册记录匹配](https://doi.org/10.1145/3690624.3709205) |
| 55 | Wang, X. and Shen, H.W. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f72d4fdfd5eb425cd81df9fe6272a533-Abstract-Conference.html) |
| 56 | Wiegreffe, S. and Pinter, Y. (2019) | 元数据一致 | [已读取正式页面](https://doi.org/10.18653/v1/D19-1002) |
| 57 | Wu, Q. et al. (2025) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v267/wu25ag.html) |
| 58 | Yao, Z., Florescu, I. and Lee, C. (2024) | 元数据一致 | [已读取正式页面](https://doi.org/10.1609/icaps.v34i1.31529) |
| 59 | Yates, R.D. et al. (2021) | 元数据一致 | [DOI 重定向至 IEEE；页面验证受限](https://doi.org/10.1109/JSAC.2021.3065072) |
| 60 | Ying, Z. et al. (2019) | 首作者缩写按官方记录修正 | [已读取正式页面](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html) |
| 61 | Yu, C. et al. (2022) | 元数据一致 | [已改为可读取的正式论文集页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html) |
| 62 | Yu, Z. and Gao, H. (2025) | 元数据一致 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html) |
| 63 | Yuan, H. et al. (2021) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v139/yuan21c.html) |
| 64 | Yuan, H. et al. (2023) | 题名/作者/DOI 一致；最终卷期待直接复核 | [DOI 重定向至 IEEE；页面验证受限](https://doi.org/10.1109/TPAMI.2022.3204236) |
| 65 | Zhang, J., Chen, Z., Mei, H., Da, L., Luo, D. and Wei, H. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/909f526db5127f8bd8158af32d9e313a-Abstract-Conference.html) |
| 66 | Zheng, X., Shirani, F., Wang, T., Cheng, W., Chen, Z., Chen, H., Wei, H. and Luo, D. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/34293d684b1012ed45c3274b4a7edc00-Abstract-Conference.html) |
| 67 | Zhou, H., Huang, A., Azizzadenesheli, K., Childers, D. and Lipton, Z. (2024) | 元数据一致 | [已读取正式页面](https://proceedings.mlr.press/v238/zhou24c.html) |

## 导出检查

Word/PDF 已同步更新，全文 73 页。67 条参考文献均有对应的 PDF 外部链接；页码映射稳定。数据表和公式与本轮修改前一致。参考文献条目设置为整段不跨页，更新页已完成视觉检查。

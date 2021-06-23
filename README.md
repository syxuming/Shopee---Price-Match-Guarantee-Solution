# 比赛介绍
Shopee是东南亚和台湾的领先电子商务平台，本次主办多模型商品匹配竞赛。使用深度学习和传统机器学习的结合分析图像和文本信息，以比较相似度，预测哪些商品是相同的产品。

# 比赛trick。
1. 多模态。
比赛会同时用到CV和NLP的模型，需要参赛者两者都比较了解。

2. 测试数据远多于训练数据。
比赛训练数据约35,000张，私有测试数据约有70,000张。这样的分布会产生一个问题，无法建立有效的验证集，导致无法使用CV得分正确估计LB得分。因为70,000张数据中会出现更多的图片相似但不属于同一类别的情况，所以在infer阶段，LB得分对相似度匹配的阈值极其敏感，需要多次提交来确认。

# 解决方案思路
我们采用三个策略融合的方式。
1.CNN模型（主导），匹配商品图像，使用了nfnet和swin_large_patch4_window12_384双模型ensemble。
2.tfidf模型，匹配商品标题，embedding=25,000。
3.图像phash值，匹配完全相同的图像。
最后对三策略的所有匹配项目去重合并（并集）得到最后的匹配方案。


# 代码、数据集、额外lib
1.代码
代码有train和inference。train是训练阶段的代码，inference是推理阶段的代码。
将train出来的模型，上传的kaggle，再在kaggle上使用inference代码即可得出结果。
代码提供ipynb版和py版，两者一致

2.数据集
本次比赛没有使用外部数据集，请至kaggle官网下载相关数据。
https://www.kaggle.com/c/shopee-product-matching/data

3.额外lib
请讲lib文件夹下的zip文件都解压到train.ipynb中libdir下，以便运行。
> **word2Vec中size参数的选择**(引自Mr.Gao)
>
> 维度的确定并不是有一个固定的值，主要的衡量指标是，如果我们知道这个里边的单词数目比较多，那么维度就相应要大一些，例如 google训练的向量，就需要 500 维度。 如果是自己的小文本，就可以是几十维度。
>
> 1. 如果训练出来的单词，我们获取的 most_similar 和这个单词几乎没什么关系，说明我们的维度设置的太大了，因为维度太大，数据量还不足够让我们的向量“调整”到我们需要的空间位置。
> 2. 如果训练出来的单词，我们的获得的 most_similar 和这个单词有关系，但是获得的 most_similar 单词都是这个单词前后左右的，例如输入“美丽”，本来应该是出现“漂亮”，但是输出的是“的人”。 这个时候很可能是我们的维度太小了，因为我们的数据太多，维度太小，没有把不同的单词拉开空间距离的差距。

#### Q：gensim的word2vec模型训练完后本身就带有了most_similar的方法去寻找某个词语的相似词语。为何还需要get_related_words这个函数，使用most_similar方法去层层扩展相似词语的相似词语。

> A：most_similar 是通过机器学习方式获得的词向量以及词向量之间的距离衡量的“相似”词汇。 但是这个相似词汇首先是从**直觉**上来讲，一个单词只有5 - 10个单词和这个单词的意思解决； 另外，从理论上来讲，我们词向量学到的词汇，most_similar 其实是**位置**相同的单词，但是位置相同只能加强**意思**相同。这两者之间是加强关系，不是充分必要的关系。 所以，我们例如 most_similar 能够获得 5 - 10 个同义词的这个现象，结合图搜索算法，获得了更多的同义词 

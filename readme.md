### AI_NLP
#### Week 1 
 **内容**
 * [x] 搜索 dfs,bfs
 * [x] 基于规则 
 * [x] 基于模式匹配的对话机器人实现（中英文，正则）
#### Week 2  
 **内容**
 * [x] 语言模型,概率模型，1-gram, 2-gram, N-gram 
 * [x] 简单爬虫 re.sub re.compile re.findall
 * [x]  维基百科语料库的语言模型
    * extractor维基百科语料库
    * 读入分析语料 
    * 繁简中文转换 HanziConv.toSimplified()
    * OOV问题-good-turning，对生词可以避免0的概率
 #### Week 3  
 **内容**
 * [x] 分析泰坦尼克号的数据（分析数据之间的关系）
    * 分析某数据的sub bins问题
 * [x] 利用一次函数拟合数据的三种方法，（随机，方向，导数）
 * [x] 利用中国城市做一个简易的导航地图 （搜索问题）
    * networkx显示中文问题
    * 双函数
    * 用BFS获得最佳路径（基于最少换程，最多换程，最短距离）

* [x] 北京地铁换程方案
  * 爬取北京地铁站信息（BS，基于css得到内容）
  * 显示所有地铁线路的graph，美化 散射（ pos = nx.spring_layout(subway_graph)）
  * 分析数据，利用bfs获得最佳换程（最短换乘，最短乘车时间），获得满足条件的所有路线
  * search_strategy(自定义排序方法)

#### Week 4

**内容**

* [x] 通过cut问题理解动态规划--不断查表过程
   - 分析子问题的重复性
   - 子问题进行存储
   - solution进行解析
* [x] python面向函数，装饰器的作用
* [x] 字符串的Edit distance--一个字符串到另一个字符串的变化--solution
* [ ] k- person-salesman (待改进，用动态规划)
* [x] 练习python web，准备项目一  <https://bottlepy.org/>

#### Week 5

**内容**

* [x] word2vec  size，min_count, workers的参数选择

* [ ] 关键词， NER
* [ ] name entity(哈工大LTP) + dependency parsing

#### Week 6

**内容**

* [x] 搜索树dfs（相似的词，例如“说”的所有同义词）

  * 读取语料库，word2Vec-获取词向量，找到同义词

* [x] TF-IDF(关键词)

  * TF（word在当前文档中出现的次数）和IDF（在多少文章中出现了word的概率的倒数）的概念
  * 获取语料库中文章的关键字

  * **文本的tf-idf向量化** fit_transform，获得输入文的所有的向量字典，这是一个矩阵，每一行代表该文本的所有词向量，而这些词向量又同样为每个文本的词典，可能为0即表示没有这个词，而每行每列记录的就是该词在该行文本中的关键程度tfidf
  * **文本的相似度对比**，获得所有文本中跟目标文本最相似的文本（排序） sim(d1,d2) = cos，距离小的相似度更高
  *  基于布尔搜索的 **搜索引擎**，输入的词在哪些文章中出现过（每个词的出现文章集合的交集）
  * 传入关键字，找到有这些关键字的文章并使用markdown加粗标记显示

* [x] WordCloud

  * 加载中文支持字体，利用上述tfidf得到文章生成的{key: score}字典关键词，生成词云
  * 添加自己图片生成词云

* [x] PageRank，自带函数，模拟网站被链接的rank




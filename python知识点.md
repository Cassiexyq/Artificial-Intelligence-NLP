### python知识点

* global用法：定义全局变量，使放在函数块中的变量在函数块外也能调用

* chr()和ord()：前者输入一个整数返回ascii符合，后者返回单个字符的ascii或者unicode数值

* hstack()和vstack()的区别: hstack()是把俩个数组水平组合，一行的拼一起；等同于
  <pre><code> concatenate((a,b),axix=1) </pre></code>
  vstack()是把第二个拼在第一个后面，垂直组合，等同于
  <pre><code> concentrate((a,b),axix=0) </pre></code>
  要注意这个数组是同维度的，即等行等列
  vstack()和平常两个数组的append()是相同的，如果对两个长度不一样的进行拼接，用np.array()

* 排序：sort()和sorted(),sort()是在原位重新排序列表，只应用再list上，sorted()是生成一个新的列表,
  key的效率高于cmp(因为key只调用一次，而cmp调用多次)，能用在所有可迭代对象上
  <pre><code>sorted(arr, key: lambda x: x[1]  # 根据第二个排序 
  sorted(arr, cmp = lambda x,y: cmp(x[1],y[1])) # 根据第二个排序 </pre></code>

* itertools模块：[博客](http://www.wklken.me/posts/2013/08/20/python-extra-itertools.html)
  [博客](http://funhacks.net/2017/02/13/itertools/)
  很强大，很高效 [具体参见代码使用](https://github.com/Cassiexyq/Program-Exercise/blob/master/SEOJ/%E5%87%BD%E6%95%B0%E4%BD%BF%E7%94%A8/itertools.py)

* 正则表达式：

  * <code>re.compile(r'正则表达式').findall(src，pos,endpos)</code> 在src中查找是否有符合该正则表达式的字串
  * <code>re.compile(r'正则表达式').match(src，pos,endpos)</code> 用正则去匹配src的开头
  * <code>re.compile(r'正则表达式').fullmatch(src)</code> 用正则去匹配整个src，即默认就是头和尾
  * <code>re.match(re.comile(r'正则表达式')，src)</code>该功能跟上面的一样，不同的是，这个传入的是数组，可以放切片数组，上面的可以传入位置参数
  * <code>re.fullmatch(re.comile(r'正则表达式')，src)</code>该功能跟上面的一样，不同的是，这个传入的是数组，可以放切片数组，上面的可以传入位置参数
  * <code>re.sub(r'正则表达式'，新的, 被替换的)</code>该功能可以去用正则去查找要被替换的，返回被替换的新的字符串
  * ^匹配开头，$匹配结尾，[^]表示里面的范围不包含，*表示对上一个对象可以>=0的个数，？表示对上一个对象0或1，+表示对上一个对象可以>=1
  * 在子表达式中开头描述范围不要用[]，会报错。
  * 匹配中文[\u4e00-\u9fa5]，匹配空格\s，匹配数组\d
  * 对一个字符串某个特点字符的查找，<code>str.find()</code>返回这个字串出现的起始位置

* 替换函数：
  * strip():这不是真正意义上的字符串，可以删除位于首位的字符
  * 对象.replace(rgExp, replceText, max): 前两个都一定要有，max可选，rgExp表示是string对象；replaceText就是一个string对象；max是一个数字，表示对象里面每一个rgExp都替换成replaceText,从左至右最多max次， replace不能用正则
  * re.sub(pattern,repl,string,count,flags):前三个必选，后两个可选，pattern表示正则表达式的模式字符串，repl是被替换的字符串，string是要被处理的原字符串，count是匹配的次数

* reduce(function, iterable[, initializer])函数：对参数序列元素进行累积如<code>reduce(add, [1,2,3,4,5])</code> 

  ​	<code>reduce(lambda x,y: x+y, [1,2,3,4,5])</code>

* eval()函数：

  ```python
  x = 7 
  eval('3 * x')  # 21
  eval('pow(2,2)') # 4 
  ```

* 用python导出数据库数据到csv加入表头信息，用sql查询得不到表头，所以插入的信息就只是每行数据，一般用csv.writer(f, dialect='excel')获得，这样就没有表头了，所以自己加一条write.writerow当作一行数据写进csv，一般如果是自己定义的数据，就可以用字典数组，用csv,DictWriter()，这就可以加表头

* 报错：expected string or bytes-like object ，出现在正则表达式，主要意思是数据类型不匹配，也就是说<code> re.findall(r'正则', sentence)</code>，其中的sentence必须是str刑，可以加str()就行了。

* 数组压平,对于就一层的，可以用sum(list, [])方法，或者chain（*list）方法，复杂一点有多层可以遍历判断是否为list，是就extend否就append，或者变成str替换 [ 和 ] 

* 输出f-string比 %-format形式和str.format()形式更快，python3.6以上支持

* lru-cache 会出现unhashable type的错，必须是可迭代对象，指向地址要不变

* word2vec: <https://rare-technologies.com/word2vec-tutorial/>

* 矩阵的计算（numpy）：

  * arange（10）获得一维10列, reshape（4，5）把一维变4行5列,

  *  zeros((4,5)) , ones((4,5))

  * np.where:

    * 只给where()函数传递一个数组作为参数，返回其中非0 元素的下标，对于二维数组，返回两个分别表示行下标和列下标的数组

    * 传入一个包含true/false作为参数。返回true的下标

       ``` python
      np.where(x%2==1) #返回数组x中所有奇数的下标
      y[np.where(y>2)] # 返回数组Y中所有大于2 的元素
       ```

    * 给where传递一个条件数组和两个值或数组，对于条件数组中等价于true的位置，从第一个值或数组张进行替换，否则从第二个中替换

      ```python
      x = np.array([1,0,3],[4,5,0])
      np.where(x, [30,40,50],[60,70,80])  # [30,70,50],[30,40,80]
      ```

* 数组

  <code>X[random_choose].toarray().shape # (1,1000)  </code>

  <code>X[random_choose].toarray()[0].shape #(1000,)</code>

* defaultdict用法：

  > ```python
  > strings = ('puppy', 'kitten', 'puppy', 'puppy',
  >            'weasel', 'puppy', 'kitten', 'puppy')
  > counts = {}
  > for kw in strings:
  >     counts[kw] += 1strings = ('puppy', 'kitten', 'puppy', 'puppy',
  >            'weasel', 'puppy', 'kitten', 'puppy')
  > counts = {}
  > for kw in strings:
  >     counts[kw] += 1
  > ```

  这是会报错的，因为每次对存在的键的数字加1，那没有键的数字就会报错，这种错误首先想到的是加入一个判断语句，判断是否在字典中出现，也可以用dicr.setdefault()方法设置默认值`dict.setdefault()`方法接收两个参数，第一个参数是健的名称，第二个参数是默认值。假如字典中不存在给定的键，则返回参数中提供的默认值；反之，则返回字典中保存的值。利用`dict.setdefault()`方法的返回值可以重写for循环中的代码，使其更加简洁。

  使用collections.defaultdict方法本身提供了默认值的功能，defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值

  ```python
  def zero(): return 0
  dd=defaultdice(zero)
  counts = defaultdict(lambda: 0) #使用lambda来定义简单的函数
  ```






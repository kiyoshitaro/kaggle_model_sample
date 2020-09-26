from pyspark import SparkContext

# # Stop spark if it existed.
# try:
#     sc.stop()
# except:
#     print('sc have not yet created!')
    
# sc = SparkContext(master = "local", appName = "First app")
# # Check spark context
# print(sc)
# # Check spark context version
# print(sc.version)

# CREATE SPARK CORE (define RDD)
sc = SparkContext.getOrCreate()
print(sc)
# Build a RDD
# nums= sc.parallelize([1,2,3,4])			
# DF_ppl = sqlContext.createDataFrame(ppl)

from pyspark.sql import SparkSession
my_spark = SparkSession.builder.getOrCreate()
print(my_spark)

# List all table exist in spark sesion
my_spark.catalog.listTables()

# spark DataFrame
train_df = my_spark.read.csv('train.csv', header = True)
train_df.show(5)
train_df.printSchema()



# train_df = pd.read_csv('houseprice/train.csv')
# y = train_df.SalePrice
# train_df.drop(["SalePrice"], axis=1, inplace=True)
# test_df = pd.read_csv('houseprice/test.csv')
# idx = test_df["Id"]
# df = pd.concat([train_df, test_df],ignore_index=True)
# df.drop(["Id"], axis=1, inplace=True)


# Để đưa dự liệu từ local lên cluster 
# chúng ta cần save nó dưới dạng một temporary table
train_df.createOrReplaceTempView('traindf_temp')
# check list all table available on catalog
my_spark.catalog.listTables()


# biến đổi dữ liệu của spark DataFrame

# .withColumn(“newColumnName”, formular): Thêm một trường mới 
# vào một bảng sẵn có. Gồm 2 tham số chính, tham số thứ nhất 
# là tên trường mới, tham số thứ 2 là công thức cập nhật tên trường. 
# Lưu ý rằng spark DataFrame là một dạng dữ liệu immutable (không thể modified được). 
# Do đó ta không thể inplace update (như các hàm fillna() hoặc replace() của pandas dataframe) 
# mà cần phải gán gi
train_df = train_df.withColumn('GarageArea_new', train_df.GarageArea/60)
train_df.printSchema()


# .withColumnRenamed(“oldColumnName”, “newColumnName”): Đổi tên của một column name trong pandas DataFrame.


train_df.drop('education_num').columns

# .select(“column1”, “column2”, … , “columnt”, formular): Lựa chọn danh sách các trường 
# trong spark DataFrame thông qua các tên column được truyền vào đưới dạng string và 
# tạo ra một trường mới thông qua formular. Lưu ý để đặt tên cho trường mới ứng với 
# formular chúng ta sẽ cần sử dụng hàm formula.alias("columnName").
avg_car = train_df.select("GarageCars", "GarageArea", (train_df.GarageCars/train_df.GarageArea).alias("avg_car"))



# .filter(condition): Lọc một bảng theo một điều kiện nào đó. 
# Condition có thể làm một string expression biểu diễn công thức lọc hoặc 
# một công thức giữa các trường trong spark DataFrame. 
# Lưu ý Condition phải trả về một trường dạng Boolean type

filter_SEA_ANC = train_df.filter("LotShape == 'IR1'") \
                        .filter("LotConfig == 'FR2'")

# .groupBy(“column1”, “column2”,…,”columnt”): Tương tự như lệnh GROUP BY của SQL, 
# lệnh này sẽ nhóm các biến theo các dimension được truyền vào groupBy. 
# Theo sau lệnh groupBy() là một build-in function của spark DataFrame được sử dụng 
# để tính toán theo một biến đo lường nào đó chẳng hạn như hàm avg(), min(), max(), sum(). 
# Tham số được truyền vào các hàm này chính là tên biến đo lường.

avg_time_org_airport = train_df.groupBy("LotShape").avg("GarageArea_new")


# .join(tableName, on = “columnNameJoin”, how = “leftouter”): Join 2 bảng với nhau 
# tương tự như lệnh left join trong SQL. Kết quả trả về sẽ là các trường mới trong bảng 
# tableName kết hợp với các trường cũ trong bảng gốc thông qua key là. 
# Lưu ý rằng columnNameJoin phải trùng nhau giữa 2 bảng.


# các lệnh biến đổi dữ liệu của spark.sql()
flights_10 = my_spark.sql('SELECT * FROM traindf_temp WHERE TotalBsmtSF > 1000')


# TRANSFORMATIONS
# Return a new RDD by applying a function to each element of this RDD
x = sc.parallelize(["b", "a", "c"]) 
y = x.map(lambda z: (z, 1)) 
print(x.collect()) 
print(y.collect())



# Return a new RDD containing only the elements that satisfy a predicate
x = sc.parallelize([1,2,3])
y = x.filter(lambda x: x%2 == 1) #keep odd values print(x.collect())
print(y.collect())



# Return a new RDD by first applying a function to all elements of this RDD, 
# and then flattening the results
# flatMap(f, preservesPartitioning=False)
x = sc.parallelize([1,2,3])
y = x.flatMap(lambda x: (x, x*100, 42)) 
print(x.collect())
print(y.collect())


# Group the data in the original RDD. Create pairs where the key is the output of
# a user function, and the value is all items for which the function yields this key.
x = sc.parallelize(['John', 'Fred', 'Anna', 'James']) 
y = x.groupBy(lambda w: w[0])
print([(k, list(v)) for (k, v) in y.collect()])


# Group the values for each key in the original RDD. Create a new pair where the
# original key corresponds to this collected group of values.
x = sc.parallelize([('B',5),('B',4),('A',3),('A',2),('A',1)]) 
y = x.groupByKey()
print(x.collect())
print(list((j[0], list(j[1])) for j in y.collect()))



# REDUCEBYKEY
words  = ["one", "two", "two", "three", "three", "three"]
wordPairsRDD = sc.parallelize(words).map(lambda word : (word, 1))
wordCountsWithReduce = wordPairsRDD .reduceByKey("+").collect()



# MAPPARTITIONSWITHINDEX
# Return a new RDD by applying a function to each partition of this RDD
x = sc.parallelize([1,2,3,4], 2)

y = x.map(lambda x: sum(x))
# vs
def f(iterator): yield sum(iterator); yield 42 
y = x.mapPartitions(f)
# glom() flattens elements on the same partition
print(y.glom().collect())



# MAPPARTITIONSWITHINDEX

x = sc.parallelize([1,2,3], 2)
def f(partitionIndex, iterator): yield (partitionIndex, sum(iterator)) 
y = x.mapPartitionsWithIndex(f)
# glom() flattens elements on the same partition 
print(x.glom().collect()) 
print(y.glom().collect())


# SAMPLE
# Return a new RDD containing a statistical sample of the original RDD
x = sc.parallelize([1, 2, 3, 4, 5]) 
y = x.sample(False, 0.4, 42) 
print(x.collect()) 
print(y.collect())


# UNION
x = sc.parallelize([1,2,3], 2) 
y = sc.parallelize([3,4], 1) 
z = x.union(y) 
print(z.glom().collect())

# JOIN
x = sc.parallelize([("a", 1), ("b", 2)])
y = sc.parallelize([("a", 3), ("a", 4), ("b", 5)]) 
z = x.join(y)
print(z.collect())


# DISTINCT
x = sc.parallelize([1,2,3,3,4]) 
y = x.distinct()

# COALESCE
# Return a new RDD which is reduced to a smaller number of partitions
x = sc.parallelize([1, 2, 3, 4, 5], 3)
y = x.coalesce(2)
print(y.glom().collect())


# KEYBY
# Create a Pair RDD, forming one pair for each item in the original RDD. The
# pair’s key is calculated from the value via a user-supplied function.
x = sc.parallelize(['John', 'Fred', 'Anna', 'James']) 
y = x.keyBy(lambda w: w[0])
print(y.collect())


# PARTITIONBY
# Return a new RDD with the specified number of partitions, 
# placing original items into the partition returned by a user supplied function
# partitionBy(numPartitions, partitioner=portable_hash)
x = sc.parallelize([('J','James'),('F','Fred'), ('A','Anna'),('J','John')], 3)
y = x.partitionBy(2, lambda w: 0 if w[0] < 'H' else 1) 
print(x.glom().collect())
print(y.glom().collect())


# ZIP
# Return a new RDD containing pairs whose key is the item in the original RDD, 
# and whose value is that item’s corresponding element (same partition, same index) 
# in a second RDD
x = sc.parallelize([1, 2, 3,4]) 
y = x.map(lambda n:n*n)
z = x.zip(y)
print(z.collect())



# ACTIONS

# GETNUMPARTITIONS
# Return the number of partitions in RDD
x = sc.parallelize([1,2,3], 2) 
y = x.getNumPartitions()
print(x.glom().collect()) 
print(y)

# COLLECT
# Return all items in the RDD to the driver in a single list


# REDUCE
# Aggregate all the elements of the RDD by applying a user function
# pairwise to elements and partial results, and returns a result to the driver
x = sc.parallelize([1,2,3,4]) 
y = x.reduce(lambda a,b: a+b)
print(x.collect()) 
print(y)


# AGGREGATE
# Aggregate all the elements of the RDD by:
# - applying a user function to combine elements with user-supplied objects, 
# - then combining those user-defined results via a second user function,
# - and finally returning a result to the driver.

seqOp = lambda data, item: (data[0] + [item], data[1] + item) 
combOp = lambda d1, d2: (d1[0] + d2[0], d1[1] + d2[1])
x = sc.parallelize([1,2,3,4])
y = x.aggregate(([], 0), seqOp, combOp)
print(y)

# MAX, SUM, MEAN, STDEV
x = sc.parallelize([2,4,1]) 
y = x.max()
print(x.collect()) 
print(y)

# COUNTBYKEY
# Return a map of keys and counts of their occurrences in the RDD
x = sc.parallelize([('J', 'James'), ('F','Fred'), ('A','Anna'), ('J','John')])
y = x.countByKey() 
print(y)


# SAVEASTEXTFILE
x = sc.parallelize([2,4,2,5],3) 
x.saveAsTextFile("./test")

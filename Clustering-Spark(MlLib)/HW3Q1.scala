import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
// Load and parse the data
val data = sc.textFile("/FileStore/tables/itemusermat")
val parsedData = data.map(s => Vectors.dense(s.split(" ",2)(1).split(" ").map(_.toDouble))).cache()     //remove movies_id from a splitted array for training purpose
val parsedData2 = data.map(s => (s.split(" ",2)(0),Vectors.dense(s.split(" ",2)(1).split(" ").map(_.toDouble)))).cache()     //make pair (movies_id,corresponding_vector)   for testing purpose

//parsedData.take(1)

// Cluster the data into two classes using KMeans
val numClusters = 10
val numIterations = 20
val seed=1
val clusters = KMeans.train(parsedData, numClusters, numIterations)    //training the model
val clusterIdxAndPoint = parsedData2.map(p =>(p._1,clusters.predict(p._2))).sortBy(_._2,false)    //making pair as (movie_id,cluster_id) and sort it on the basis of cluster_id
val cluster0=clusterIdxAndPoint.filter(x=>(x._2==0)).take(5)
val cluster1=clusterIdxAndPoint.filter(x=>(x._2==1)).take(5)
val cluster2=clusterIdxAndPoint.filter(x=>(x._2==2)).take(5)
val cluster3=clusterIdxAndPoint.filter(x=>(x._2==3)).take(5)
val cluster4=clusterIdxAndPoint.filter(x=>(x._2==4)).take(5)
val cluster5=clusterIdxAndPoint.filter(x=>(x._2==5)).take(5)
val cluster6=clusterIdxAndPoint.filter(x=>(x._2==6)).take(5)
val cluster7=clusterIdxAndPoint.filter(x=>(x._2==7)).take(5)
val cluster8=clusterIdxAndPoint.filter(x=>(x._2==8)).take(5)
val cluster9=clusterIdxAndPoint.filter(x=>(x._2==9)).take(5)

val final_united_cluster=cluster0.union(cluster1).union(cluster2).union(cluster3).union(cluster4).union(cluster5).union(cluster6).union(cluster7).union(cluster8).union(cluster9)

val final_united_cluster_rdd=sc.parallelize(final_united_cluster)

val movies_data = sc.textFile("/FileStore/tables/movies.dat")
val parsedMoviesData = movies_data.map(s =>(s.split("::",2)(0),s.split("::",2)(1))).cache()     //make pair as (movies_id,title&Genre) from a splitted array for joining purpose

val final_res=final_united_cluster_rdd.join(parsedMoviesData).sortBy(_._2._1,true).map(x=>("Cluster:"+x._2._1+"\tMovie_Id:"+x._1+"\tTitle:"+x._2._2.split("::")(0)+"\t Genre:"+x._2._2.split("::")(1)))
//val predictions = clusters.predict(parsedData)     // testing the model
//printing the results
//predictions.take(10)
final_res.collect.foreach(println)

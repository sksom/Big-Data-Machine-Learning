import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
// Load and parse the data
val data = sc.textFile("/FileStore/tables/ratings.dat")
val parsedData = data.map(s =>(s.split("::")(0).toInt,s.split("::")(1).toInt,s.split("::")(2).toDouble)).map(s=>Rating(s._1,s._2,s._3)).cache()     //remove timestamp from a splitted array
//parsedData.take(2)
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val train_data = splits(0)
val test_data = splits(1)
val rank = 10
val numIterations = 20
val model = ALS.train(train_data, rank, numIterations,0.01)
val known_label = test_data.map(s=>((s.user, s.product), s.rating))
val formatted_test_data = test_data.map(s=>(s.user, s.product))
val predictions = model.predict(formatted_test_data).map(s=>((s.user, s.product), s.rating))
//predictions.take(2)
val final_res=predictions.join(known_label)
//final_res.take(2)
val MSE = final_res.map(s=> (math.pow((s._2._1 - s._2._2),2))).mean()
println("Mean squared error:"+MSE)

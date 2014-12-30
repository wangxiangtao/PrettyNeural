package com.prettymatch.core.mlpscala

import java.util.Arrays
import java.util.HashSet
import java.util.Random
import java.util.Set
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import scala.collection.JavaConversions._
import com.google.common.util.concurrent.FutureCallback
import com.google.common.util.concurrent.Futures
import com.google.common.util.concurrent.ListeningExecutorService
import com.google.common.util.concurrent.MoreExecutors
import weka.classifiers.Classifier
import weka.classifiers.RandomizableClassifier
import weka.core.ConjugateGradientOptimization
import weka.core.Instance
import weka.core.Instances
import weka.core.Optimization
import weka.core.RevisionUtils
import weka.core.Utils
import weka.filters.Filter
import weka.filters.unsupervised.attribute.NominalToBinary
import weka.filters.unsupervised.attribute.RemoveUseless
import weka.filters.unsupervised.attribute.ReplaceMissingValues
import weka.filters.unsupervised.attribute.Standardize
import java.util.concurrent.TimeUnit

class MLP extends RandomizableClassifier  {

  class OptBFGS extends Optimization {
    override def objectiveFunction( x : Array[Double]) : Double = {
      m_MLPParameters = x
      return calculateSE()
    }
    override def evaluateGradient(x: Array[Double]) : Array[Double] ={
      m_MLPParameters = x 
      return calculateGradient()
    }
    override def getRevision() = RevisionUtils.extract("Xiangtao version1")
  }
  
  class OptCGD extends ConjugateGradientOptimization {
    override def objectiveFunction( x : Array[Double]) : Double = {
      m_MLPParameters = x
      return calculateSE()
    }
    override def evaluateGradient(x: Array[Double]) : Array[Double] ={
      m_MLPParameters = x 
      return calculateGradient()
    }
    override def getRevision() = RevisionUtils.extract("Xiangtao version1")
  }

  // The number of hidden units
  var numHiddenUnit = 2

  // The class index of the dataset
  var m_classIndex = -1

  // A reference to the actual data
  var m_data :Instances = _

  // The number of classes in the data
  var m_numClasses = -1

  // The number of attributes in the data , include class
  var m_numAttributes = -1

  // The parameter vector
  var m_MLPParameters : Array[Double] = _

  // Offset for output unit parameters
  var OFFSET_WEIGHTS = -1

  // Offset for parameters of hidden units
  var OFFSET_ATTRIBUTE_WEIGHTS = -1

  // The ridge parameter
  var m_ridge = 0.01

  // Whether to use conjugate gradient descent rather than BFGS updates
  var m_useCGD = false

  // Tolerance parameter for delta values
  var m_tolerance = 1.0e-6

  // The number of threads to use to calculate gradient and squared error
  var m_numThreads = 8

  // The size of the thread pool
  var m_poolSize = 2

  // The standardization filer = _
  var  m_Filter: Filter = _

  // An attribute filter
  var  m_AttFilter : RemoveUseless = _

  // The filter used to make attributes numeric.
  var  m_NominalToBinary: NominalToBinary = _

  // The filter used to get rid of missing values.
  var  m_ReplaceMissingValues: ReplaceMissingValues = _

  // a ZeroR model in case no model can be built from the data
  var  m_ZeroR: Classifier = _

  // Thread pool
  var executorService : ListeningExecutorService = _

  def initializeClassifier(instances: Instances) : Instances = {

    // can classifier handle the data?
    getCapabilities().testWithFail(instances)

    var data = new Instances(instances)
    data.deleteWithMissingClass()

    // Make sure data is shuffled
    var random = new Random(m_Seed)
    if (data.numInstances() > 1) {
      random = data.getRandomNumberGenerator(m_Seed)
    }
    data.randomize(random)

    // Replace missing values
    m_ReplaceMissingValues = new ReplaceMissingValues()
    m_ReplaceMissingValues.setInputFormat(data)
    data = Filter.useFilter(data, m_ReplaceMissingValues)

    // Remove useless attributes
    m_AttFilter = new RemoveUseless()
    m_AttFilter.setInputFormat(data)
    data = Filter.useFilter(data, m_AttFilter)

    // only class? -> build ZeroR model
    if (data.numAttributes() == 1) {
      System.err
        .println("Cannot build model (only class attribute present in data after removing useless attributes!), "
          + "using ZeroR model instead!")
      m_ZeroR = new weka.classifiers.rules.ZeroR()
      m_ZeroR.buildClassifier(data)
      return null
    } else {
      m_ZeroR = null
    }

    // Transform nominal attributes
    m_NominalToBinary = new NominalToBinary
    m_NominalToBinary.setInputFormat(data)
    data = Filter.useFilter(data, m_NominalToBinary)

    // Standardize data
    m_Filter = new Standardize
    m_Filter.setInputFormat(data)
    data = Filter.useFilter(data, m_Filter)

    m_classIndex = data.classIndex
    m_numClasses = data.numClasses
    m_numAttributes = data.numAttributes

    OFFSET_WEIGHTS = 0
    // For each output unit, there are m_numUnits + 1 inputs.  1 means bias 
    // so for output layer ,  there are total (m_numUnits + 1) * m_numClasses weights
    OFFSET_ATTRIBUTE_WEIGHTS = (numHiddenUnit + 1) * m_numClasses 
    
    // Here m_numAttributes = (input units +1) since it include class
    // For each hidden unit, there are (number of input units + 1) weights.  
    m_MLPParameters = Array.fill[Double](OFFSET_ATTRIBUTE_WEIGHTS + numHiddenUnit * m_numAttributes)
                      { 0.1 * random.nextGaussian()}
    return data
  }

  override def buildClassifier( data: Instances) : Unit = {
    m_data = initializeClassifier(data)
    if (m_data == null)  return
    
    executorService = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(m_numThreads));

    var opt : Optimization = null
    opt = if (!m_useCGD) new OptBFGS  else new OptCGD
    opt.setDebug(m_Debug)

    // No constraints
    val constraints = Array.fill[Double](2, m_MLPParameters.length){Double.NaN} 

    m_MLPParameters = opt.findArgmin(m_MLPParameters, constraints)
    while (m_MLPParameters == null) {
      m_MLPParameters = opt.getVarbValues()
      if (m_Debug) {
        System.out.println("First set of iterations finished, not enough!")
      }
      m_MLPParameters = opt.findArgmin(m_MLPParameters, constraints)
    }
    if (m_Debug) {
      System.out.println("SE (normalized space) after optimization: "
        + opt.getMinFunction())
    }

    m_data = new Instances(m_data, 0) // Save memory
    
    executorService.shutdown();
    while (!executorService.isTerminated()) {
        executorService.awaitTermination(10, TimeUnit.SECONDS);
    }
    executorService = null  // if not ,the class can not be serialized
  }

  /**
   * Calculates the (penalized) squared error based on the current parameter vector.
   */
  def  calculateSE() : Double = {
    // Set up result set, and chunk size
    val chunksize = m_data.numInstances() / m_numThreads
    val results = new HashSet[Future[Double]]()

    for (j <- 0 until m_numThreads) {

      val low = j * chunksize
      val high = if(j < m_numThreads - 1)  (low + chunksize) else m_data.numInstances()
  
        // Create and submit new job, where each instance in batch is processed
      val futureSE = executorService.submit(new Callable[Double]() {
         override def call() : Double = {
           var hiddenOutputs = Array.ofDim[Double](numHiddenUnit)
              var SE: Double = 0
              for (k <- low until high) {
                val inst = m_data.instance(k)
                // Calculate necessary input/output values and error term
                hiddenOutputs = getHiddenLayerOutputs(inst, hiddenOutputs, null)
                // For all class values
                for ( i <- 0 until m_numClasses) {
                  // Get target (make them slightly different from 0/1 for better convergence)
                  // If current class equal with current output unit , then 1 ,otherwise 0  
                  val target = if( inst.value(m_classIndex) == i)  0.99 else 0.01  ////// not sure
                  // Add to squared error
                  val err = target - getOutputLayerOutput(i, hiddenOutputs, null)
                  SE += inst.weight() * err * err
                }
              }
              return SE
          }
       });
       results.add(futureSE)
    }

    // Calculate SE
    var SE : Double = 0
    results.foreach { f => SE += f.get }
    
    // Calculate sum of squared weights, excluding bias
    var squaredSumOfWeights : Double = 0
    for ( i <- 0 until m_numClasses) {
      var offsetOW = OFFSET_WEIGHTS + (i * (numHiddenUnit + 1))
      for ( k <- 0 until numHiddenUnit) {
        squaredSumOfWeights += m_MLPParameters(offsetOW + k) * m_MLPParameters(offsetOW + k)
      }
    }
    for (k <- 0 until numHiddenUnit) {
      val offsetW = OFFSET_ATTRIBUTE_WEIGHTS + k * m_numAttributes
      for (j <- 0 until m_classIndex) {
        squaredSumOfWeights += m_MLPParameters(offsetW + j) * m_MLPParameters(offsetW + j)
      }
      for (j <- m_classIndex + 1 until m_numAttributes) {
        squaredSumOfWeights += m_MLPParameters(offsetW + j) * m_MLPParameters(offsetW + j)
      }
    }

    return ((m_ridge * squaredSumOfWeights) + (0.5 * SE)) / m_data.sumOfWeights()
  }

  def calculateGradient() : Array[Double] = {
    val chunksize = m_data.numInstances() / m_numThreads
    val results = new HashSet[Future[Array[Double]]]()
    for (j <- 0 until m_numThreads) {
      val low = j * chunksize
      val high = if(j < m_numThreads - 1)  (low + chunksize) else m_data.numInstances()
      var futureGrad = executorService.submit(new Callable[Array[Double]]() {
        override def call() : Array[Double] = {
          val outputs = Array.ofDim[Double](numHiddenUnit)
          val deltaHidden = Array.ofDim[Double](numHiddenUnit)
          val sigmoidDerivativeOutput = Array.ofDim[Double](1)
          val sigmoidDerivativesHidden = Array.ofDim[Double](numHiddenUnit)
          val localGrad = Array.ofDim[Double](m_MLPParameters.length)
          for (k <- low until high) {
            val inst = m_data.instance(k)
            getHiddenLayerOutputs(inst, outputs, sigmoidDerivativesHidden)
            updateGradientInOutputLayer(localGrad, inst, outputs, sigmoidDerivativeOutput, deltaHidden)
            updateGradientInHiddenLayer(localGrad, inst, sigmoidDerivativesHidden, deltaHidden)
          }
          return localGrad
        }
      })
      results.add(futureGrad)
    }

    // Calculate final gradient
    val grad = Array.ofDim[Double](m_MLPParameters.length)
    results.foreach ( futureGrad => {
        var lg = futureGrad.get()
        for (i<- 0 until lg.length)  grad(i) += lg(i)
    })

    // For all network weights, perform weight decay
    for (i <- 0 until m_numClasses) {
      val offsetOW = OFFSET_WEIGHTS + (i * (numHiddenUnit + 1))
      for (k <- 0 until numHiddenUnit) {
        grad(offsetOW + k) += m_ridge * 2 * m_MLPParameters(offsetOW + k)
      }
    }
    for (k <- 0 until numHiddenUnit) {
      val offsetW = OFFSET_ATTRIBUTE_WEIGHTS + k * m_numAttributes
      for (j <- 0 until m_classIndex) {
        grad(offsetW + j) += m_ridge * 2 * m_MLPParameters(offsetW + j)
      }
      for (j <- m_classIndex + 1 until m_numAttributes ) {
        grad(offsetW + j) += m_ridge * 2 * m_MLPParameters(offsetW + j)
      }
    }

    val factor = 1.0 / m_data.sumOfWeights()
    for (i <- 0 until grad.length) {
      grad(i) *= factor
    }
    return grad
  }
  
  /**
   * Calculates the array of outputs of the hidden units. Also calculates
   * derivatives if d != null.
   */
  def  getHiddenLayerOutputs(inst: Instance, hiddenOutputs:Array[Double], derivatives: Array[Double]) : Array[Double]= {
    for (i <- 0 until numHiddenUnit) {
      val offsetW = OFFSET_ATTRIBUTE_WEIGHTS + i * m_numAttributes
      var sum : Double = 0
      for (j <- 0 until m_classIndex) {
        sum += inst.value(j) * m_MLPParameters(offsetW + j)
      }
      sum += m_MLPParameters(offsetW + m_classIndex)
      for (j <- m_classIndex + 1 until m_numAttributes) {
        sum += inst.value(j) * m_MLPParameters(offsetW + j)
      }
      hiddenOutputs(i) = sigmoid(sum, derivatives, i)
    }
    return hiddenOutputs
  }

  /**
   * Update the gradient for the weights in the output layer.
   */
  def updateGradientInOutputLayer(grad: Array[Double], inst: Instance, outputs: Array[Double],
    sigmoidDerivativeOutput: Array[Double], deltaHidden: Array[Double]): Unit = {

    // Initialise deltaHidden
    Arrays.fill(deltaHidden, 0.0)

    // For all output units
    for (j <- 0 until m_numClasses) {
      // Get output from output unit j
      val pred = getOutputLayerOutput(j, outputs, sigmoidDerivativeOutput)
      // Get target (make them slightly different from 0/1 for better
      // convergence)
      val target = if (inst.value(m_classIndex).toInt == j) 0.99 else 0.01
      // Calculate delta from output unit
      val deltaOut = inst.weight() * (pred - target) * sigmoidDerivativeOutput(0)
      // Go to next output unit if update too small
      if (!(deltaOut <= m_tolerance && deltaOut >= -m_tolerance)) {
        // Establish offset
        val offsetOW = OFFSET_WEIGHTS + (j * (numHiddenUnit + 1))
        // Update deltaHidden
        for (i <- 0 until numHiddenUnit) {
          deltaHidden(i) += deltaOut * m_MLPParameters(offsetOW + i)
        }
        // Update gradient for output weights
        for (i <- 0 until numHiddenUnit) {
          grad(offsetOW + i) += deltaOut * outputs(i)
        }
        // Update gradient for bias
        grad(offsetOW + numHiddenUnit) += deltaOut
      }
    }
  }

  def updateGradientInHiddenLayer(grad:Array[Double],  inst:Instance,
    sigmoidDerivativesHidden: Array[Double], deltaHidden: Array[Double]) {
    // Finalize deltaHidden
    for (i <- 0 until numHiddenUnit) {
      deltaHidden(i) *= sigmoidDerivativesHidden(i)
    }
    // Update gradient for hidden units
    for (i <- 0 until numHiddenUnit) {
      // Skip calculations if update too small
      if (!(deltaHidden(i) <= m_tolerance && deltaHidden(i) >= -m_tolerance)) {
           // Update gradient for all weights, including bias at classIndex
        var offsetW = OFFSET_ATTRIBUTE_WEIGHTS + i * m_numAttributes
        for (l <- 0 until m_classIndex) {
          grad(offsetW + l) += deltaHidden(i) * inst.value(l)
        }
        grad(offsetW + m_classIndex) += deltaHidden(i)
        for (l <- m_classIndex + 1 until m_numAttributes) {
          grad(offsetW + l) += deltaHidden(i) * inst.value(l)
        }
      }
    }
  }

  /**
   * Calculates the output of output unit based on the given hidden layer
   * outputs. Also calculates the derivative if d != null.
   */
  def getOutputLayerOutput(indexOfOutputUnit: Int,hiddenOutputs: Array[Double],derivatives: Array[Double]) : Double = {
    var offsetOW = OFFSET_WEIGHTS + (indexOfOutputUnit * (numHiddenUnit + 1))
    var result : Double = 0
    for (i <- 0 until numHiddenUnit) {
      result += m_MLPParameters(offsetOW + i) * hiddenOutputs(i)
    }
    result += m_MLPParameters(offsetOW + numHiddenUnit)
    return sigmoid(result, derivatives, 0)
  }

  /**
   * Computes approximate sigmoid function. Derivative is stored in second
   * argument at given index if d != null.
   * Compute approximate sigmoid ,  1/1+e^(-x)
   * e^x = lim(1+x/n)^n  ,  n ==> finite
   * e^(-x) =   lim(1 - x/n)^n
   * reference http://ybeernet.blogspot.sg/2011/03/speeding-up-sigmoid-function-by.html
   */ 
   def sigmoid(x: Double, d: Array[Double], index: Int) : Double = {
    val y = 1.0 - x / 8192.0
    val z = Math.pow(y, 8192)
    val output = 1.0 / (1.0 + z)
    if (d != null) {
  //    d[index] = output * (1.0 - output) / y
        d(index) = output * (1.0 - output)
    }
    return output
  }

  @Override
  override def distributionForInstance( instance: Instance) : Array[Double] = {
    m_ReplaceMissingValues.input(instance)
    var inst = m_ReplaceMissingValues.output()
    m_AttFilter.input(inst)
    inst = m_AttFilter.output()

    // default model?
    if (m_ZeroR != null) {
      return m_ZeroR.distributionForInstance(inst)
    }
    m_NominalToBinary.input(inst)
    inst = m_NominalToBinary.output()
    m_Filter.input(inst)
    inst = m_Filter.output()
    val dist = Array.ofDim[Double](m_numClasses)
    var outputs = Array.ofDim[Double](numHiddenUnit)
    outputs = getHiddenLayerOutputs(inst, outputs, null)
    for (i <- 0 until m_numClasses) {
      dist(i) = getOutputLayerOutput(i, outputs, null).toDouble
      dist(i) = dist(i) match {
                  case x if x < 0 => 0
                  case y if y > 1 => 1
                  case _ =>  dist(i)
                }
    }
    Utils.normalize(dist)
    return dist
  }

  override def toString() : String = {
     if (m_ZeroR != null) return m_ZeroR.toString()
     if (m_MLPParameters == null) return "Classifier not built yet"
     var s = "MLPClassifier with ridge value " + m_ridge + " and " +
             numHiddenUnit + " hidden units (useCGD=" + m_useCGD + ")\n\n"
     for (i <- 0 until numHiddenUnit) {
        for (j <- 0 until m_numClasses) {
          s += "Output unit " + j + " weight for hidden unit " + i + ": " + 
          m_MLPParameters(OFFSET_WEIGHTS + j * (numHiddenUnit + 1) + i).toString() + "\n"
        }
        s += "\nHidden unit " + i + " weights:\n\n"
        for (j <- 0 until m_numAttributes) {
          if (j != m_classIndex) {
            s += m_MLPParameters(OFFSET_ATTRIBUTE_WEIGHTS + (i * m_numAttributes) + j) +
               " " + m_data.attribute(j).name() + "\n"
          }
        }
        s += "\nHidden unit " + i+ " bias: " + m_MLPParameters(OFFSET_ATTRIBUTE_WEIGHTS  +
            (i * m_numAttributes + m_classIndex)) + "\n\n"
     }
     for (j <- 0 until m_numClasses) {
        s += "Output unit " + j + " bias: "+  m_MLPParameters(OFFSET_WEIGHTS + j * (numHiddenUnit + 1) + numHiddenUnit) + "\n"
     }
     return s
  }
}

package com.prettyneural.trainer

import java.util.ArrayList
import java.util.List
import com.prettyneural.core.Layer
import com.prettyneural.core.LayerFactory
import com.prettyneural.network.StackedRBM

class StackedRBMTrainer {
    var  stackedRBM:StackedRBM = null
    var  inputTrainer:SimpleRBMTrainer = null
    var  momentum:Float = 0
    var  l2:Float = 0
    var  targetSparsity:Float = 0 
    var  learningRate:Float = 0
    var  layerFactory:LayerFactory = null
    
     def this( stackedRBM:StackedRBM, momentum:Float, l2:Float, targetSparsity:Float, learningRate:Float, layerFactory:LayerFactory )
    {
      
        this()
        
        this.stackedRBM = stackedRBM;
        this.momentum = momentum;
        this.l2 = l2;
        this.targetSparsity = targetSparsity;
        this.learningRate = learningRate;
        this.layerFactory = layerFactory;
        inputTrainer = new SimpleRBMTrainer(momentum, l2, targetSparsity, learningRate, layerFactory );
    }

    def  setLearningRate(newRate: Float){
        learningRate = newRate;
        inputTrainer.learningRate = newRate;
    }

    //Starts at the bottom of the DBN and uses the output of one RBM as the input of
    //the next.  This continues till it hits stopAt.  Then it trains the RBM with the
    //mutated input batch.  It also allows a second batch to be appended to a input batch
    //So you can combine a deep RBM feature with a second input.
    //
    //An example being features of a digit picture combined with the digit label.
    def learn( bottomBatch:List[Layer], topBatch:List[Layer],stopAt: Int): Double =
    {
        if (topBatch != null && !topBatch.isEmpty() && topBatch.size() != bottomBatch.size())
            throw new IllegalArgumentException("TopBatch != BottomBatch")

        if (stopAt < 0 || stopAt > stackedRBM.innerRBMs.size())
            throw new IllegalArgumentException("Invalid stopAt")


        val nextInputs = new ArrayList[Layer](bottomBatch)

        for (i <- 0 until stopAt)
        {
            
            //At stopping point do actual learning
            if (i == stopAt-1)
            {
                return inputTrainer.learn(stackedRBM.innerRBMs.get(i), nextInputs, false);
            }

            //Use the hidden of this layer as the inputs of the next layer
            for (j <- 0 until nextInputs.size())
            {
                var next = stackedRBM.innerRBMs.get(i).activateHidden(nextInputs.get(j),null);

                if (topBatch != null && !topBatch.isEmpty() && i == stopAt - 2)
                {
                    val nextConcat = new Array[Float](next.size()+topBatch.get(j).size());
                    System.arraycopy(next.get(),0,nextConcat,0,next.size());
                    System.arraycopy(topBatch.get(j).get(), 0, nextConcat, next.size(), topBatch.get(j).size());

                    next = layerFactory.create(nextConcat);
                }

                nextInputs.set(j,next);
            }
        }

        throw new AssertionError("Didn't find a level top stop at");
    }
}
package com.prettyneural.trainer

import java.util.List
import scala.collection.JavaConversions._
import com.prettyneural.network.SimpleRBM
import com.prettyneural.core.LayerFactory
import com.prettyneural.core.Layer

class SimpleRBMTrainer {
    var  momentum: Float = 0
    var l2: Float = 0
    var  targetSparsity: Float = 0
    var  learningRate: Float = 0
    var  layerFactory: LayerFactory = null

    var gw : Array[Layer]= null
    var gv: Layer = null
    var gh: Layer = null
    
      def this( momentum: Float, l2: Float, targetSparsity: Float, learningRate: Float,  layerFactory: LayerFactory)
    {
        this()
        this.momentum = momentum;
        this.l2 = l2;
        this.targetSparsity = targetSparsity;
        this.learningRate = learningRate;
        this.layerFactory = layerFactory;
    }

    def learn(  rbm: SimpleRBM ,inputBatch:  List[Layer] , reverse: Boolean): Double =
    {
        var batchsize = inputBatch.size;

        if (gw == null || gw.length != rbm.biasHidden.size() || gw(0).size() != rbm.biasVisible.size())
        {
            gw = new Array[Layer](rbm.biasHidden.size);
            for(i <- 0 until gw.length)
                gw(i) = layerFactory.create(rbm.biasVisible.size());

            gv = layerFactory.create(rbm.biasVisible.size());
            gh = layerFactory.create(rbm.biasHidden.size());
        }
        else
        {
            for(i <- 0 until gw.length)
                gw(i).clear;
            gv.clear;
            gh.clear;
        }
        
        // Contrastive Divergance
       inputBatch.foreach( input => {
              val it = if(reverse) rbm.reverseIterator(input) else rbm.iterator(input);

              var t1 = it.next();    //UP
              var t2 = it.next();    //Down

              for (i <- 0 until gw.length)
                  for (j <- 0 until gw(i).size)
                      gw(i).add(j, (t1.hidden.get(i) * t1.visible.get(j)) - (t2.hidden.get(i) * t2.visible.get(j)));

              for (i <- 0 until gv.size)
                  gv.add(i, t1.visible.get(i) - t2.visible.get(i));

              for (i <- 0 until gh.size)
                  gh.add(i,  if(targetSparsity == 0) t1.hidden.get(i) - t2.hidden.get(i) else targetSparsity - t1.hidden.get(i));

       })


        // Average
        for (i <- 0 until gw.length)
        {
            for (j <- 0 until gw(i).size())
            {
                gw(i).div(j, batchsize);
                gw(i).mult(j, 1 - momentum);
                gw(i).add(j,  momentum * (gw(i).get(j) - l2*rbm.weights(i).get(j)));
                
                rbm.weights(i).add(j, learningRate * gw(i).get(j));
            }
        }

        var error : Double = 0.0;

        for (i <- 0 until gv.size)
        {
            gv.div(i, batchsize);

            error += Math.pow(gv.get(i), 2);

            gv.mult(i, 1 - momentum);
            gv.add(i, momentum * (gv.get(i) * rbm.biasVisible.get(i)));

            rbm.biasVisible.add(i, learningRate * gv.get(i));
        }

        error = Math.sqrt(error/gv.size());

        if (targetSparsity != 0)
        {
            for (i <- 0 until gh.size)
            {
                gh.div(i,batchsize);
                gh.set(i, targetSparsity - gh.get(i));
            }
        }
        else
        {
            for (i <- 0 until gh.size)
            {
                gh.div(i, batchsize);

                gh.mult(i, 1 - momentum);
                gh.add(i, momentum * (gh.get(i) * rbm.biasHidden.get(i)));

                rbm.biasHidden.add(i, learningRate * gh.get(i));
            }
        }

        return error;
    }
}
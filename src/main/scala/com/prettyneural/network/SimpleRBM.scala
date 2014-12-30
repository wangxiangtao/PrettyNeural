package com.prettyneural.network

import java.util.Random
import java.io.IOException
import java.io.DataOutput
import java.util.Arrays
import java.io.DataInput
import java.util.Iterator
import com.prettyneural.core.Layer
import com.prettyneural.core.LayerFactory
import com.prettyneural.core.Tuple
import com.prettyneural.core.Utilities

class SimpleRBM {
    var biasVisible : Layer  = null
    var biasHidden : Layer  = null
    var weights : Array[Layer] = null
    var lfactory: LayerFactory = null
    var rand: Random = null
    var scale = 0.001f
    var gaussianVisibles = false
    
    def this(numVisible: Int,numHidden: Int, gaussianVisibles: Boolean, lfactory: LayerFactory){
       this()
        rand = new Random();
        if (SimpleRBM.randomSeed !=  0)
            rand.setSeed(SimpleRBM.randomSeed);

        this.lfactory = lfactory;

        this.gaussianVisibles = gaussianVisibles;

        // initialize nodes
        biasVisible = lfactory.create(numVisible);
         for (i <- 0 until numVisible)
            biasVisible.set(i, (scale * rand.nextGaussian()).toFloat)

        biasHidden = lfactory.create(numHidden)
        for (i <- 0 until numHidden)
            biasHidden.set(i, (scale * rand.nextGaussian()).toFloat)

        // initialize weights and weight change matrices
        weights = new Array[Layer](numHidden)

        // randomly initialize weights
        for(i <- 0 until numHidden) {
            weights(i) = lfactory.create(numVisible)
            for (j <- 0 until numVisible)
                weights(i).set(j, (2 * scale * rand.nextGaussian()).toFloat)
        }
    }
    
   def save( dataOutput: DataOutput)  {
        dataOutput.write(LayerFactory.MAGIC);

        dataOutput.writeBoolean(gaussianVisibles);
        lfactory.save(biasVisible, dataOutput);
        lfactory.save(biasHidden, dataOutput);

        for (i <- 0 until weights.length)
            lfactory.save(weights(i),dataOutput);
    }

    def  load( dataInput: DataInput,  lfactory:LayerFactory )  {

        this.lfactory = lfactory;

        var magic = new Array[Byte](4);
        dataInput.readFully(magic);

        if (!Arrays.equals(LayerFactory.MAGIC, magic))
            throw new IOException("Bad File Format");

        gaussianVisibles = dataInput.readBoolean();

        biasVisible = lfactory.load(dataInput);
        biasHidden = lfactory.load(dataInput);
        weights = new Array[Layer](biasHidden.size());

        for (i <- 0 until weights.length)
            weights(i) = lfactory.load(dataInput);
    }

    // Given visible data, return the expected hidden unit values.
    def  activateHidden( visible:Layer, bias: Layer) : Layer =
    {
        val workingHidden = lfactory.create(biasHidden.size());

        if (visible.size() != biasVisible.size())
            throw new IllegalArgumentException("Mismatched input "+visible.size()+" != "+biasVisible.size());


        if (bias != null && workingHidden.size() != bias.size() && bias.size() > 1)
            throw new AssertionError("bias must be 0,1 or hidden length");


        // dot product of weights and visible
        for (i <- 0 until weights.length)
            for (k <- 0 until visible.size())
                workingHidden.add(i, weights(i).get(k) * visible.get(k));

        //Add hidden bias
        for (i <- 0 until workingHidden.size()) {
            var inputBias = 0.0f;

            if (bias != null && bias.size() != 0)
                inputBias = if(bias.size() == 1 ) bias.get(0) else bias.get(i);

            workingHidden.set(i, Utilities.sigmoid(workingHidden.get(i) + biasHidden.get(i) + inputBias));
        }

        return workingHidden;
    }

    // Given hidden states, return the expected visible unit values.
    def  activateVisible( hidden: Layer, bias: Layer): Layer =
    {
        val workingVisible =  lfactory.create(biasVisible.size());

        if (bias != null && workingVisible.size() != bias.size() && bias.size() > 1)
            throw new AssertionError("bias must be 0,1 or visible length");

        // dot product of weights and hidden
        for (k <- 0 until weights.length)
            for (i <- 0 until workingVisible.size())
                workingVisible.add(i, weights(k).get(i) * hidden.get(k));

        //Add visible bias
        for (i <- 0 until workingVisible.size())
        {
            workingVisible.add(i, biasVisible.get(i));

            //Add input bias (if any)
            if (bias != null && bias.size() != 0)
                workingVisible.add(i, if(bias.size() == 1 ) bias.get(0) else bias.get(i));

            if (!gaussianVisibles)
                workingVisible.set(i, Utilities.sigmoid(workingVisible.get(i)));
        }

        return workingVisible;
    }

    def  iterator( visible: Layer) : Iterator[Tuple]  = {
        return iterator(visible, new Tuple.Factory(visible));
    }

    def  reverseIterator( visible: Layer) : Iterator[Tuple] = {
        return reverseIterator(visible, new Tuple.Factory(visible));
    }

    def iterator(  visible:Layer, tfactory:Tuple.Factory) : Iterator[Tuple] = 
    {
        return new Iterator[Tuple]()
        {
            var v = visible;
            var h = activateHidden(v,null);

            def  hasNext(): Boolean = 
            {
                return true;
            }

            def  next(): Tuple =
            {
                var t = tfactory.create(v, h);

                // Next updown
                v = activateVisible(Utilities.bernoulli(h),null);
                h = activateHidden(v,null);

                return t;
            }

            def  remove()
            {

            }
        };
    }

    def  reverseIterator(  hidden: Layer , tfactory: Tuple.Factory): Iterator[Tuple] = 
    {
        return new Iterator[Tuple]()
        {
            var v = activateVisible(Utilities.bernoulli(hidden),null);
            var h = hidden;


            def  hasNext() : Boolean =
            {
                return true;
            }

            def  next() : Tuple =
            {
                var t = tfactory.create(v, h);

                // Next downup
                v = activateVisible(Utilities.bernoulli(h),null);
                h = activateHidden(v,null);

                return t;
            }

            def  remove()
            {

            }
        };
    }

    def freeEnergy() : Float = {
        var energy = 0.0f;

        for (j <- 0 until biasHidden.size())
            for (i <- 0 until biasVisible.size())
                energy -= biasVisible.get(i) * biasHidden.get(j) * weights(j).get(i);

        return energy;
    }

}

object SimpleRBM {
    var randomSeed: Long = 0
   
}
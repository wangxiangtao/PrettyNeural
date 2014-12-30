package com.prettyneural.core

import java.util.Random

object Utilities {

    val  staticRand = new Random()
    
    def  mean(input: Layer) : Float = 
    {
        var m : Float = 0.0f;
        for (i <- 0 until input.size())
            m += input.get(i)
        m/input.size()
    }

    def  stddev( input: Layer,mean: Float) : Float = 
    {
        var sum = 0.0f;
        for (i<- 0 until input.size())
            sum += Math.pow(input.get(i) - mean, 2).toFloat

        Math.sqrt(sum/(input.size() - 1)).toFloat
    }

    def sigmoid(x: Float) = (1.0f / (1.0f + Math.exp(-x))).toFloat

    def  bernoulli( input: Layer): Layer = 
    {
        val output = input.cloneLayer
        //using uniform distribution, filter out all negative values
        //from inputs, keeping mostly strong weights
        for (i <- 0 until output.size())
            output.set(i, if(staticRand.nextFloat() < input.get(i) ) 1.0f else 0.0f);

        return output;
    }
}
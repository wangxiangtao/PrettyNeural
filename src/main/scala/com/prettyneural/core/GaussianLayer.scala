package com.prettyneural.core

class GaussianLayer(de: Array[Float]) extends Layer(de) {
//  y[0
    var delegate : Layer = null
    var mean : Float = 0
    var stddev: Float = 0
    
     def this(delegate: Layer)
    {
        this(delegate.layer)
        this.delegate = delegate
        convertToStddev();
    } 
    
    def this(delegate: Layer, base: Layer)
    {
        this(delegate.layer)
        val gbase : GaussianLayer = new GaussianLayer(base)
        mean = gbase.mean;
        stddev = gbase.stddev;
    }

    def convertToStddev()
    {
        mean = Utilities.mean(delegate);
        stddev = Utilities.stddev(delegate, mean);
        stddev = if( stddev < 0.1f ) 0.1f else stddev;

        var min = Double.MaxValue
        var max = Double.MinValue

        for (i <- 0 until delegate.size())
        {
            var v = (delegate.get(i) - mean)/stddev;
            if (v > max) max = v;
            if (v < min) min = v;

            delegate.set(i, v);
        }

    }

    def fromGaussian() : Array[Float] = {
        var min = Double.MaxValue
        var max = Double.MinValue

        var output = new Array[Float](delegate.size())
        for (i <- 0 until output.length) {
            var v =  delegate.get(i);

            //Squash > 2 sigma
            if (Math.abs(v) > 2)
                v /= 2;

            v  = v * stddev + mean;

            if (v > max) max = v;
            if (v < min) min = v;

            output(i) = if(v < 0) 0 else v
            output(i) = if(v > 255) 255 else v
        }
        return output;
    }
    
    override def set(i:Int,f: Float) {
        delegate.set(i,f);
    }

    override def get(i:Int):Float =  {
        delegate.get(i);
    }

    override def add(i: Int,f:Float) {
        delegate.add(i,f);
    }

    override def div(i: Int,f:Float) {
        delegate.div(i,f);
    }

    override def mult( i:Int, f:Float) {
        delegate.div(i,f);
    }

    override def size(): Int = {
         delegate.size;
    }

    override def cloneLayer(): Layer = {
        delegate.cloneLayer;
    }

   override def clear() {
        delegate.clear;
    }

    override def copy(src: Array[Float]) {
        delegate.copy(src);
    }

    override def get() : Array[Float] =  {
        delegate.get();
    }

}
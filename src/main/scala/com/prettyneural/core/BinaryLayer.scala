package com.prettyneural.core

class BinaryLayer(de: Array[Float]) extends Layer(de) {
//  y[0
    var delegate : Layer = null
    
     def this(delegate: Layer)
    {
        this(delegate.layer)
        this.delegate = delegate
        convertToBinary()
    } 
    
    def convertToBinary()
    {
        for ( i <- 0 until size())
        {
            val v = get(i)
            set(i, if(v > 30) 1.0f else 0.0f);
        }
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

object BinaryLayer {
  
    def fromBinary( delegate: Layer): Array[Float] =  {
        var output = new Array[Float](delegate.size);
        for (i <- 0 until output.length) {
            output(i) = delegate.get(i) * 255.0f;
        }
        output
    }
}
package com.prettyneural.network

import java.util.ArrayList
import java.util.List
import sun.reflect.generics.reflectiveObjects.NotImplementedException
import java.util.Iterator
import scala.collection.JavaConversions._
import java.io.DataOutput
import java.io.IOException
import java.util.Arrays
import java.io.DataInput
import com.prettyneural.core.LayerFactory
import com.prettyneural.core.Tuple
import com.prettyneural.core.Layer

class StackedRBM(s: String) extends SimpleRBM{
     var layerFactory:LayerFactory = null
     var layerSizes:List[Integer] = null
     var customInputSizes:List[Integer] = null
     var gaussianFlag:List[Boolean] = null
     var innerRBMs: List[SimpleRBM] = null
     
    def this()
    {   
        this("Default value from auxiliary constructor")
        layerSizes = new ArrayList[Integer]();
        customInputSizes = new ArrayList[Integer]();
        gaussianFlag = new ArrayList[Boolean]();
        innerRBMs = new ArrayList[SimpleRBM]();
    }

    def setLayerFactory( layerFactory: LayerFactory): StackedRBM =   {
        this.layerFactory = layerFactory;
        return this;
    }

    def  addLayer( numUnits: Int,gaussian: Boolean): StackedRBM = {
        if (!innerRBMs.isEmpty())
            throw new RuntimeException("Can't add new layers after already built");
        layerSizes.add(numUnits);
        gaussianFlag.add(gaussian);
        return this;
    }
    
    def  withCustomInput( numUnits:Int): StackedRBM = {
        while (customInputSizes.size() < layerSizes.size())
            customInputSizes.add(null);
        customInputSizes.set(customInputSizes.size()-1,numUnits);
        return this;
    }

    def  build(): StackedRBM =  {
        if (!innerRBMs.isEmpty())
            return this; //already built
        if (layerSizes.size() <= 1)
            throw new IllegalArgumentException("Requires at least two layers to build");
        for (i <- 0 until layerSizes.size()-1)
        {
            var inputSize = layerSizes.get(i);
            if (!customInputSizes.isEmpty() && customInputSizes.size() >= i && customInputSizes.get(i+1) != null)
                inputSize = customInputSizes.get(i+1);
            innerRBMs.add(new SimpleRBM(inputSize, layerSizes.get(i+1), gaussianFlag.get(i), layerFactory));
            System.err.println("Added RBM "+inputSize+ " -> "+layerSizes.get(i+1));
        }
        return this;
    }

    override def activateHidden( visible: Layer, bias: Layer) : Layer = {
        throw new NotImplementedException();
    }

    override def activateVisible( hidden: Layer, bias: Layer) : Layer = {
        throw new NotImplementedException();
    }

    override def  iterator( visible: Layer) : Iterator[Tuple] ={
       println(" this method have some issue, please fix below issue")
        var input = visible;

        var stackNum = innerRBMs.size();

        for (i <- 0 until stackNum)
        {
            val iRBM = innerRBMs.get(i);
            if (i == (stackNum-1))
            {
                return iRBM.iterator(visible,new Tuple.Factory(input));
            }
            println("below line is the issue, ")
//            visible = iRBM.activateHidden(visible , null);
        }

        throw new AssertionError("code bug");
    }

    override def  reverseIterator(visible: Layer) : Iterator[Tuple] = {
        throw new NotImplementedException();
    }

//    override def iterator( visible:Layer,  tfactory: Tuple) : Iterator[Tuple] = {
//        throw new NotImplementedException();
//    }
//
//    override def  reverseIterator(visible: Layer, tfactory: Tuple ) : Iterator[Tuple] = {
//        throw new NotImplementedException();
//    }

    override def  save( dataOutput: DataOutput) {

        dataOutput.write(LayerFactory.MAGIC);

        dataOutput.writeInt(innerRBMs.size());

        innerRBMs.foreach { rbm => rbm.save(dataOutput) }
          
    }

    override def load( dataInput: DataInput, layerFactory: LayerFactory) {

        this.layerFactory = layerFactory;

        val magic = new Array[Byte](4);
        dataInput.readFully(magic);

        if (!Arrays.equals(LayerFactory.MAGIC, magic))
            throw new IOException("Bad File Format");

        var numInner = dataInput.readInt();

        for (i <- 0 until numInner)
        {
            System.err.println("Loading rbm "+i);

            val loaded = new SimpleRBM();
            loaded.load(dataInput, layerFactory);
            innerRBMs.add(loaded);
        }
    }

    def  getInnerRBMs() = innerRBMs

    override def freeEnergy(): Float = {
       var energy = 0.0f;

       innerRBMs.foreach( rbm => energy += rbm.freeEnergy() )
           
       return energy;
    }
}
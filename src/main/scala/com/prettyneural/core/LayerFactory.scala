package com.prettyneural.core

import java.awt.image.BufferedImage
import java.io.IOException
import java.io.DataOutput
import java.util.Arrays
import java.io.DataInput

object LayerFactory{
   val MAGIC = Array[Byte](0xf0.toByte, 0x0d.toByte, 0x00.toByte, 0x0F.toByte)
}
class LayerFactory {
    
    def create(size: Int) = new Layer(size)

    def create(start: Array[Float]) = new Layer(start)

    def create(img: BufferedImage) : Layer = {
        val layer = create(img.getWidth() * img.getHeight())
        var width = 0
        var height = 0
        for (i <- 0 until layer.size()) {
            layer.set(i, img.getData().getSample(width, height, 0))
            width += 1
            if (width >= img.getWidth()) {
                width = 0;
                height += 1
            }
        }
        layer
    }


    def save(layer: Layer,dataOutput: DataOutput) = {
        //First write magic #
        dataOutput.write(LayerFactory.MAGIC)
        var floats = layer.get()
        if (floats.length != layer.size())
            throw new IOException("get().length != size()")
        //Number of elements
        dataOutput.writeInt(layer.size())
        for (i <- 0 until floats.length )
            dataOutput.writeFloat(floats(i))
    }

    def load(dataInput: DataInput) : Layer = {
        var magic = new Array[Byte](4)
        dataInput.readFully(magic)
        if (!Arrays.equals(LayerFactory.MAGIC, magic))
            throw new IOException("Bad File Format")
        var size = dataInput.readInt()
        if (size < 0)
            throw new IOException("Invalid size")
        var input = new Array[Float](size)
        for (i <- 0 until size)
            input(i) = dataInput.readFloat()
        return create(input)
    }
}
package com.prettyneural.core

class Tuple() {
    var  visible: Layer = null
    var  hidden: Layer = null
    var  input: Layer = null  //For a DBN this is the initial input layer

    def this( input:Layer,visible:Layer , hidden:Layer)
    {
        this()
        this.input = input;
        this.visible = visible;
        this.hidden = hidden;
    }
   
}

object Tuple{
    class Factory {
        var input: Layer = null
        def this( input: Layer) {
          this()
            this.input = input;
        }
        def create( visible: Layer,  hidden: Layer): Tuple =
        {
            return new Tuple(input,visible,hidden);
        }
    }
}
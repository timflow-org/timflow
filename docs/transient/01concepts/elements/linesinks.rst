Line-sinks
==========

Line-sinks are lines along which water is taken out of or put into the aquifer. The
inflow/outflow along the line-sink does not vary along a line-sink. There are two types
of line-sinks: line-sinks for which the head is specified at the center of the
line-sink, and line-sinks for which the total discharge is specified and the head along
the line-sink is unknown but uniform.

Both types of line-sinks may have an entry resistance defined by the resistance
:math:`c` (dimension: time). The inflow :math:`\sigma_i` into the line-sink in layer
:math:`i` is a function of the head :math:`h_i` just outside the line-sink in layer
:math:`i` and the head :math:`h_{ls}` inside the line-sink:

    .. math::
        \sigma_i = w(h_i - h_{ls})/c
        
This equation is applied along each control point of the line-sink. :math:`w` is the
distance over which water infiltrates into the line-sink. The distance may be the width
of the stream, for example, in case of a partially penetrating stream. In case the
stream penetrates the aquifer layer fully, the distance may equal the thickness of the
aquifer layer (if water enters primarily from one side), or twice the aquifer thickness
(if water enters from both sides).

1. :class:`~timflow.transient.linesink.HeadLineSink` is a line-sink for which the head is specified
   along the line-sink.

2. :class:`~timflow.transient.linesink.HeadLineSinkString` is a string of head-specified line-sinks

3. :class:`~timflow.transient.linesink.LineSinkDitchString` is a string of line-sink ditches

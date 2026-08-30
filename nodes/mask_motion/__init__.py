"""Mask motion — masks that MOVE, authored by hand rather than tracked.

Not every moving mask needs a tracker. Sometimes you know where the thing goes
because you can see it, and drawing the path once is faster and more controllable
than fixing a solver's output frame by frame.

Circle today. Other shapes (rectangle, ellipse, polygon, a brush that keeps its
stroke) drop in here later as their own modules, sharing the same idea: draw a
path in the editor, get an animated MASK batch out.
"""

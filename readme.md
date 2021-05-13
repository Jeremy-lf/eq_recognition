step1: install conda environment``` conda env create -f environment.yml ```step2: activate conda env``` conda activate eq_model ```step3: run demo``` python demo.py ./test_data/ ```result:```[0/26] c1.bmpc1.bmp -> V = a b h[1/26] c10.bmpc10.bmp -> a ^ { - n } = \frac { 1 } { a ^ { n } }[2/26] c11.bmpc11.bmp -> ( \frac { b } { a } ) ^ { - n } = ( \frac { a } { b } ) ^ { n }[3/26] c12.bmpc12.bmp -> ( \sqrt { a } ) ^ { 2 } = a ( a \geqslant 0 )[4/26] c13.bmpc13.bmp -> a ^ { 0 } = 1 ( a \neq 0 )[5/26] c2.bmpc2.bmp -> ( a + b ) ( a - b ) = a ^ { 2 } - b ^ { 2 }[6/26] c3.bmpc3.bmp -> ( a + b ) ( a ^ { 2 } - a b + b ^ { 2 } ) = a ^ { 3 } + b ^ { 3 }[7/26] c4.bmpc4.bmp -> a ^ { 2 } + b ^ { 2 } = ( a + b ) ^ { 2 } - 2 a b[8/26] c5.bmpc5.bmp -> ( a \pm b ) ^ { 2 } = a ^ { 2 } \pm 2 a b + b ^ { 2 }[9/26] c6.bmpc6.bmp -> ( a - b ) ( a ^ { 2 } + a b + b ^ { 2 } ) = a ^ { 3 } - b ^ { 3 }[10/26] c7.bmpc7.bmp -> ( a - b ) ^ { 2 } = ( a + b ) ^ { 2 } - 4 a b[11/26] c8.bmpc8.bmp -> ( \frac { a } { b } ) ^ { n } = \frac { a ^ { n } } { \ln n }[12/26] c9.bmpc9.bmp -> ( a b ) ^ { n } = a ^ { n } b ^ { n }```
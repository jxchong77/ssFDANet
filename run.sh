while true
do 
    python train_one.py;
    # python train_one_2.py;
    # python test.py 3DMAD;
    # python test.py HKBUv1+;
    # python test.py casia_3d;
    python test.py casia;
    python test.py Oulu;
    python test_featuremap.py casia;
    python test_featuremap.py Oulu;

done
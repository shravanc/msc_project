	CX�%l��@CX�%l��@!CX�%l��@	�Ji^�7D?�Ji^�7D?!�Ji^�7D?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$CX�%l��@u�<��?A2����@Yĳ�?*	+���wQ@2F
Iterator::Models���M�?!Э�.�E@)��r0� �?1����:@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�&l?�?!
Z f��;@)4ڪ$��?12P�
I>9@:Preprocessing2S
Iterator::Model::ParallelMap���E��?!\���_11@)���E��?1\���_11@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[1]::Concatenate֭���7�?!�]���5@){���?1�a�v��0@:Preprocessing2X
!Iterator::Model::ParallelMap::Zipg�UId�?!�/R�L@)����jo?1J5]�U�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicej�drjgh?!�߬�@)j�drjgh?1�߬�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensora��q6]?!�N@�j@)a��q6]?1�N@�j@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�,��\n�?!��i �6@)>]ݱ�&U?1��q��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	u�<��?u�<��?!u�<��?      ��!       "      ��!       *      ��!       2	2����@2����@!2����@:      ��!       B      ��!       J	ĳ�?ĳ�?!ĳ�?R      ��!       Z	ĳ�?ĳ�?!ĳ�?JCPU_ONLY
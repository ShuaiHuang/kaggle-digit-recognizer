# `tf.contrib.data.Dataset.flat_map()`与`tf.contrib.data.Dataset.map()`的区别

TensorFlow中`flat_map()`与`map()`是两个很相似的数据集操作，搞清楚这两个概念就能在数据集处理上面少踩一些坑。

首先来看`tf.contrib.data.Dataset.flat_map()`的[说明](https://www.tensorflow.org/api_docs/python/tf/contrib/data/TextLineDataset#flat_map)：
> Maps map_func across this dataset and flattens the result.

`tf.contrib.data.Dataset.map()`的[说明](https://www.tensorflow.org/api_docs/python/tf/contrib/data/TextLineDataset#map)为：
> Maps map_func across this datset.

从函数功能上来看，`flat_map()`在对Dataset中每一个元素进行map操作后，还会对结果进行平展（flatten）操作，而`map()`仅仅只会对数据集中元素进行map操作。

其次，从函数参数角度来看，
`flat_map()`参数`map_func`是将具有嵌套结构的张量映射成一个数据集，即：
> A function mapping a nested structure of tensors (having shapes and types defined by self.output_shapes and self.output_types) to a Dataset

`map()`参数`map_func`是将具有嵌套结构的张量映射成另一个具有嵌套结构的张量，即：
> A function mapping a nested structure of tensors (having shapes and types defined by self.output_shapes and self.output_types) to another nested structure of tensors.

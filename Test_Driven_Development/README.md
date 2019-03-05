

It is more difficult to write tests for Machine learning tools because of outcome uncertainties.
Usually depends on data quality, context/human intuition. (e.g. could we assert AUROC>0.6?)

Here is a list that I experienced/gathered: 

1. Were model weights changed by training?
e.g. from: https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765

before = sess.run(tf.trainable_variables())
_ = sess.run(model.train, feed_dict={
           image: np.ones((1, 100, 100, 3)),
           })
after = sess.run(tf.trainable_variables())
for b, a, n in zip(before, after):
  # Make sure something changed.
  assert (b != a).any()
  
# You see, in tensorflow batch_norm actually has is_training defaulted to False, so adding this line of code won’t actually normalize your input during training!
  
  
2. Do processed matrix have fewer columns (and desired columns are there!), but same rows as raw matrix.

3. For comparing stats, are we using predictions for the same set of test?
Comment: This helps me a lot, since I have two versions of train_test_split, and either version was check-pointed for different site data (messy!). 
I want to make sure the comparison of performance was on the same test set, almost like a 'paired' comparison. 

4. Always handle the edge case of empty input! 
This is less relevant to ML, but could happen on up/down-streams of processing. 
For example, even though pandas could handle empty dataframe, but could not handle column names that's not present!

5. Check pandas dataframe have the same subset/order in the train/test sets. 
This brings me some trouble before using Transformer. As sklearn does not care about column names, but the order of columns.


import tensorflow as tf
import numpy as np

def fold1(f, li, init):
    (_, a1) = tf.while_loop(lambda i, a: i<tf.shape(li)[0], lambda i, a: (i+1, f(a, li[i])), (0,init))
    #note: not `lambda (i,a)`
    return a1

def fold2(f, li1, li2, init):
    (_, a1) = tf.while_loop(lambda i, a: i<tf.shape(li1)[0], lambda i, a: (i+1, f(a, li1[i], li2[i])), (0,init))
    return a1

def foldn(f, init, *lis):
    (_, a1) = tf.while_loop(lambda i, a: i<tf.shape(li1)[0], lambda i, a: (i+1, f(a, *[li[i] for li in lis])), (0,init))
    return a1

"""
easy tree function

* `f`: combining function, t -> t -> t
* `leaves_inds`: locations to write, [Int] (as tensor)
* `leaves`: values of leaves, type [t] (as tensor)
"""
def easy_tree(f, child_inds, leaves_inds, leaves, input_inds=None, inputs = None):
    #create tensor array
    init_array = tf.TensorArray(tf.float32, size=10, dynamic_size=True)
    #write (leaves_inds[i], leaves[i]) for each i
    a1 = fold2(lambda a, x, y: a.write(x, y), leaves_inds, leaves, init_array)
    
    #now combine
    def parent_step(a, child_ind):
        l_child = a.read(child_ind[0])
        r_child = a.read(child_ind[1])
        return a.write(child_ind[2], f(l_child, r_child))
    #write results of computations
    #a = tf.foldl(parent_step, child_inds, a) DOES NOT WORK because tensor arrays don't work with foldl.
    a2 = fold1(parent_step, child_inds, a1)
    return a2.read(child_inds[-1][2])
    

def test():
    #easy_tree(f, child_inds, leaves_inds, leaves, input_inds=None, inputs = None)
    t= easy_tree(tf.add, tf.constant([[0,1,3], [3,2,4]]), tf.constant([0,1,2]), tf.constant([[4.0],[6.0],[9.0]]))
    #tf.constant([0,1,2], tf.int32)
    #run session
    sess = tf.Session()
    ans = sess.run(t)
    print(ans)

if __name__=="__main__":
    test()

import tensorflow as tf
import numpy as np

# easy tree function
def easy_tree(f, child_inds, leaves_inds, leaves, input_inds=None, inputs = None):
    #leaves_inds1 = tf.expand_dims(leaves_inds, -1)
    #leaves_infos = tf.concat(1, [leaves_inds1, leaves])
    #can't do -1
    #create tensor array
    init_array = tf.TensorArray(tf.float32, size=10, dynamic_size=True)
    #write step 
    #def write_step(a, l_info):
    #    return a.write(l_info[0], l_info[1:])
    #for everything in leaves_inds, write
    #a  = tf.foldl(write_step, leaves_infos, init_array)

    (_, a1) = tf.while_loop(lambda i,a: i<tf.shape(leaves_inds)[0], lambda i,a: (i+1, a.write(leaves_inds[i], leaves[i])), (0, init_array))
    # leaves_inds[i]
    #write_step(a, leaves_infos[i])), (0,init_array))
    #note: not `lambda (i,a)`
    #return a1.read(2)
    
    #now combine
    def parent_step(a, child_ind):
        l_child = a.read(child_ind[0])
        r_child = a.read(child_ind[1])
        return a.write(child_ind[2], f(l_child, r_child))
    #write
    #a = tf.foldl(parent_step, child_inds, a)
    (_, a2) = tf.while_loop(lambda i,a: i<tf.shape(child_inds)[0], lambda i,a: (i+1, parent_step(a, child_inds[i])), (0,a1))
    return a2.read(child_inds[-1][2])
    

def test():
    """
    x = tf.constant(np.random.randn(4,4))
    i = tf.constant(3, tf.int32)
    y = x[i]
    sess = tf.Session()
    ans = sess.run(y)
    print(ans)
    """
    t= easy_tree(tf.add, tf.constant([[0,1,3], [3,2,4]]), tf.constant([0,1,2]), tf.constant([[4.0],[6.0],[9.0]]))
    #tf.constant([0,1,2], tf.int32)
    #easy_tree(f, child_inds, leaves_inds, leaves, input_inds=None, inputs = None)
    sess = tf.Session()
    ans = sess.run(t)
    print(ans)
    
    

if __name__=="__main__":
    test()

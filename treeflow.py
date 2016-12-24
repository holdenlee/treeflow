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
    (_, a1) = tf.while_loop(lambda i, a: i<tf.shape(lis[0])[0], lambda i, a: (i+1, f(a, *[li[i] for li in lis])), (0,init))
    return a1

def triplet_flatten(t):
    return (t[0], t[1][0], t[1][1])

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

"""
tree function

* `fs`: combining functions
    * If has_outputs = False, fs has type t,t,*args -> t
    * If has_outputs = True, fs has type t,t,*args -> (t, o), where o is an output at each node
    * Above examples are assuming binary tree. Can also have ternary tree t,t,t,*args -> t, etc.
    * If is_list_fn, then tree can have arbitrary number of children, and fs has type [t],*args -> t.
* `child_inds`: [child 1, child 2,..., parent index]
* `leaves_inds`: locations to write, [Int] (as tensor)
* `leaves`: values of leaves, type [t] (as tensor)
* `node_inputs`: A list of tensors *args to feed into fs, corresponding to nodes in `leaves_inds`
* `has_outputs`: Whether to output
* `is_list_fn`: Whether f takes as input a list (all the child inputs are stacked as a tensor).
"""
def treeflow(fs, child_inds, leaves_inds, leaves, node_inputs = [], has_outputs = False, is_list_fn = False, def_size=10, ans_type=tf.float32, output_type=tf.float32, degree=2):
    #create tensor array
    init_array = tf.TensorArray(ans_type, size=def_size, dynamic_size=True)
    #write (leaves_inds[i], leaves[i]) for each i
    #a1 = fold2(lambda a, x, y: a.write(x, y), leaves_inds, leaves, init_array)
    a1 = init_array.scatter(leaves_inds, leaves)
    
    #now combine
    def parent_step(a, child_ind, *extra_inputs):
        if is_list_fn:
            children = a.gather(child_ind[:-1])
            ans = fs(children, *extra_inputs)
        else:
            #'Tensor' object is not iterable.
            #children = [a.read(i) for i in child_ind[:-1]]
            #USE THIS FOLLOWING LINE FOR v0.12
            #children = tf.unstack(a.gather(child_ind[:-1]))
            children = tf.unpack(a.gather(child_ind[:-1]), num=degree)
            #return a.write(child_ind[-1], f(*(children+extra_inputs)))
            ans = fs(*(children+list(extra_inputs))) #extra_inputs is tuple
        if has_outputs:
            return (ans[1], a.write(child_ind[-1],ans[0]))
        else:
            return a.write(child_ind[-1], ans)
    #write results of computations
    #a = tf.foldl(parent_step, child_inds, a) DOES NOT WORK because tensor arrays don't work with foldl.
    def parent_step_output(i, a, o, child_ind, *extra_inputs):
        (ans, a1) = parent_step(a, child_ind, *extra_inputs)
        return (a1,o.write(i, ans))
    if has_outputs:
        output_array = tf.TensorArray(output_type, size=1, dynamic_size=True)
        #(_, a1) = tf.while_loop(lambda i, a: i<tf.shape(li1)[0], lambda i, a: (i+1, f(a, li1[i], li2[i])), (0,init))
        (_, a2, oa) = tf.while_loop(lambda i, a, o: i<tf.shape(child_inds)[0], lambda i, a, o: triplet_flatten((i+1, parent_step_output(i, a, o, child_inds[i], *[li[i] for li in node_inputs]))), (0, a1, output_array))
        return (a2.read(child_inds[-1][-1]), oa.pack())
    else:
        a2 = foldn(parent_step, a1, child_inds, *node_inputs)
        return a2.read(child_inds[-1][-1])

"""
* `fs`: scattering functions
    * If has_outputs == False, `fs` is t,*args -> tensor of t's
    * If has_outputs == True, `fs` is t,*args -> (tensor of t's, output)
"""
def treefold_unfold(fs, child_inds, root_ind, root, read_inds, node_inputs = [], has_outputs = False, def_size=10, ans_type=tf.float32, output_type=tf.float32, degree=2):
    #create tensor array
    init_array = tf.TensorArray(ans_type, size=def_size, dynamic_size=True)
    #write value to root
    a1 = init_array.write(root_ind,root)
    #propagate to children
    def branch_step(a, child_ind, *extra_inputs):
        anss = fs(a.read(child_ind[-1]),*extra_inputs)
        if has_outputs:
            a = a.scatter(child_ind[:-1], ans[0])
            return (a,ans[1])
        else:
            a = a.scatter(child_ind[:-1], ans)
            return a
    def branch_step_output(i, a, o, child_ind, *extra_inputs):
        (ans, a) = branch_step(a, child_ind, *extra_inputs)
        return (a,o.write(i, ans))
    if has_outputs:
        output_array = tf.TensorArray(output_type, size=1, dynamic_size=True)
        #(_, a1) = tf.while_loop(lambda i, a: i<tf.shape(li1)[0], lambda i, a: (i+1, f(a, li1[i], li2[i])), (0,init))
        (_, a2, oa) = tf.while_loop(lambda i, a, o: i>=0, lambda i, a, o: triplet_flatten((i-1, branch_step_output(i, a, o, child_inds[i], *[li[i] for li in node_inputs]))), (tf.shape(child_inds)[0], a1, output_array))
        return (a2.gather(read_inds), oa.pack())
    else:
        a2 = foldn(parent_step, a1, reversed(child_inds), *node_inputs)
        return a2.gather(read_inds)

def test():
    #easy_tree(f, child_inds, leaves_inds, leaves, input_inds=None, inputs = None)
    t= easy_tree(tf.add, tf.constant([[0,1,3], [3,2,4]]), tf.constant([0,1,2]), tf.constant([[4.0],[6.0],[9.0]]))
    #tf.constant([0,1,2], tf.int32)
    #run session
    sess = tf.Session()
    ans = sess.run(t)
    print(ans)
    t= treeflow(tf.add, tf.constant([[0,1,3], [3,2,4]]), tf.constant([0,1,2]), tf.constant([[4.0],[6.0],[9.0]]))
    #tf.constant([0,1,2], tf.int32)
    #run session
    sess = tf.Session()
    ans = sess.run(t)
    print(ans)
    def add_and_output(x,y,z):
        a = tf.add(x,y)
        return (a, tf.add(a,z))
    t = treeflow(add_and_output, tf.constant([[0,1,3], [3,2,4]]), tf.constant([0,1,2]), tf.constant([[4.0],[6.0],[9.0]]), node_inputs = [tf.constant([10.0,20.0])], has_outputs = True)
    #tf.constant([0,1,2], tf.int32)
    #run session
    sess = tf.Session()
    ans = sess.run(t)
    print(ans)
    #stack for v0.12
    t = treefold_unfold(lambda x: tf.pack([x/2, x/2]), tf.constant([[0,1,3], [3,2,4]]), tf.constant(4), tf.constant(8.0), tf.constant([0,1,2]))
    sess = tf.Session()
    ans = sess.run(t)
    print(ans)

if __name__=="__main__":
    test()

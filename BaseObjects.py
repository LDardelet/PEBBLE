from functools import wraps
from six import add_metaclass

def _list_method_wrapper(list_function):
    @wraps(list_function)
    def Wrapped_func(self, *args, **kwa): # self shold refer to Roster instance
        Func_res = list_function(self, *args, **kwa)
        print(list_function)
        if type(Func_res) == list:
            return self.__class__(Func_res)
        else:
            return Func_res
    return Wrapped_func

class _RosterMetaClass(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        cls_parent = bases[0]

        list_to_wrap = [(n, cls_parent) for n in cls_parent.__dict__ if (callable(getattr(cls_parent, n)) and n != '__init__' and not n in cls.__dict__.keys())]
        list_to_set = [(n, cls) for n in cls.__dict__ if (callable(getattr(cls, n)) and n != '__init__')]
        for attr, parent in list_to_wrap:
            method = getattr(parent, attr)
            setattr(cls, attr, _list_method_wrapper(method))
        for attr, parent in list_to_set:
            method = getattr(parent, attr)
            setattr(cls, attr, method)
        return cls

#@add_metaclass(_RosterMetaClass)
class roster(list):
    def __call__(self, *args, **kwargs):
        return self.__class__(item(*args, **kwargs) for item in self)
        
    def __getattribute__(self, name):
        if name[0] != '_':
            try:
                return self.__class__(item.__getattribute__(name) for item in self)
            except:
                return super(list, self).__getattribute__(name)
        elif name[2] == 'd':
            return super(self[0].__class__, self[0]).__getattribute__(name)
        else:
            return super(list, self).__getattribute__(name)

    def __add__(self, arg):
        return self.__class__(list.__add__(self, arg))

    def __mul__(self, arg):
        return self.__class__(list.__mul__(self, arg))

    def __rmul__(self, arg):
        return self.__class__(list.__rmul__(self, arg))

    def __getitem__(self, arg):
#        if type(arg) == slice:
#            print "slice"
#            return self.__class__(list.__getitem__(self, arg))
#        else:
#            print "item"
            return list.__getitem__(self, arg)

    def __getslice__(self, *args):
        return self.__class__(list.__getslice__(self, *args))

def _typedlist_method_wrapper(list_function):
    @wraps(list_function)
    def Wrapped_func(self, *args, **kwa): # self shold refer to Roster instance
        Func_res = list_function(self, *args, **kwa)
        print(list_function)
        if type(Func_res) == list:
            return self.__class__(self.__elems_type__, Func_res)
        else:
            return Func_res
    return Wrapped_func

class _TypedListMetaClass(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        cls_parent = bases[0]

        list_to_wrap = [(n, cls_parent) for n in cls_parent.__dict__ if (callable(getattr(cls_parent, n)) and n != '__init__' and not n in cls.__dict__.keys())]
        list_to_set = [(n, cls) for n in cls.__dict__ if (callable(getattr(cls, n)) and n != '__init__')]
        for attr, parent in list_to_wrap:
            method = getattr(parent, attr)
            setattr(cls, attr, _typedlist_method_wrapper(method))
        for attr, parent in list_to_set:
            method = getattr(parent, attr)
            setattr(cls, attr, method)
        return cls

#@add_metaclass(_TypedListMetaClass)
class TypedList(list):
    def __init__(self, etype, elements = []):
        list.__init__(self, elements)
        self.__elems_type__ = etype

    def __add__(self, arg):
        return self.__class__(self.__elems_type__, list.__add__(self, arg))

    def __mul__(self, arg):
        return self.__class__(self.__elems_type__, list.__mul__(self, arg))

    def __rmul__(self, arg):
        return self.__class__(self.__elems_type__, list.__rmul__(self, arg))

    def __getslice__(self, *args, **kwargs):
        return self.__class__(self.__elems_type__, list.__getslice__(self, *args, **kwargs))

    def __getitem__(self, *args, **kwargs):
        return list.__getitem__(self, *args, **kwargs)

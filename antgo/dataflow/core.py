# encoding=utf-8
# @Time    : 17-6-6
# @File    : core.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from collections import defaultdict
from functools import total_ordering
from antgo.utils import logger
import itertools
import sys

if sys.version_info[0] == 2:
  def items(dictionary):
    
    """Return a set-like object, a view on key/value pairs of a dict"""
    return dictionary.viewitems()

  def values(dictionary):
    """Return an object providing a view on the values of a dictionary"""
    return dictionary.viewvalues()

  def get_func_name(function, default=None):
    """Return the name of the function, falling back to a default"""
    return getattr(function, 'func_name', default)
else:
  def items(dictionary):
    """Return a set-like object, a view on key/value pairs of a dict"""
    return dictionary.items()

  def values(dictionary):
    """Return an object providing a view on the values of a dictionary"""
    return dictionary.values()

  def get_func_name(function, default=None):
    """Return the name of the function, falling back to a default"""
    return getattr(function, '__name__', default)


class _DIRTY(object):
  """Class definition for the dirty node special value"""

  def __str__(self):
    return 'DIRTY'

  def __repr__(self):
    return '<antgo.dataflow.DIRTY>'

  def __eq__(self, other):
    return self.__class__ == other.__class__

  def __ne__(self, other):
    return self.__class__ != other.__class__


DIRTY = _DIRTY()


class BaseNode(object):
  """Base class for Inputs and Nodes"""

  _name_counters = defaultdict(int)

  def __init__(self, name=None, value=DIRTY, auto_trigger=False):
    self.name = name or self._generate_name()
    self._value = value
    self._dependents = set()
    self._auto_trigger = auto_trigger

  def _connect(self, dependent):
    """Set the given Node as a dependent of this Node or Input

    Immediately paints the new dependent Node dirty if this Node has
    already been evaluated or if a value has already been set for this
    Input.

    Connecting Nodes always invalidates the triggered Nodes cache.

    """
    if dependent not in self._dependents:
        self._dependents.add(dependent)
        if self._value is not DIRTY:
            dependent._set_value(DIRTY)

  def _disconnect(self, dependent):
    """Remove given Node from the set of dependents of this Node or Input

    Immediately paints the new dependent Node dirty if this Node has
    previously been evaluated or if a value has previously been set for
    this Input.

    Disconnecting Nodes always invalidates the triggered nodes cache.

    """
    if dependent in self._dependents:
      self._dependents.remove(dependent)
      if self._value is not DIRTY:
        dependent._set_value(DIRTY)

  def _set_value(self, value):
    """Set a new value for this Node or Input

    If this caused the value to change, paints dependent Nodes dirty and
    returns the set of those dependent Nodes which are marked "triggered"
    and should be re-evaluated.

    When called by ``set_value`` from external code, the ``get_triggered``
    argument must be ``True`` so the return value is cached.  Internal
    calls set ``get_triggered=False`` so memory isn't wasted for caching the
    triggered dependents of intermediate Nodes.

    This private method can be used as a debugging tool to set values of
    non-input Nodes.

    """
    # test if neither, one of or both the old and the new value are DIRTY
    dirty_count = len([v for v in [value, self._value] if v is DIRTY])
    if dirty_count == 2:
      # both DIRTY, no need to touch anything
      return set()
    if dirty_count == 0 and self._value_eq(value):
      # both non-DIRTY but equal, no need to touch anything
      return set()
    # either one is DIRTY, or values aren't equal, update the value and
    # paint the dependent Nodes dirty
    self._value = value
    self._set_dependents_dirty()
    # try to trigger some auto-node
    if DIRTY != value:
      self._trigger()

  def _value_eq(self, other_value):
    return False

  def get_value(self):
    """Return the value of the object"""
    raise NotImplementedError('The get_value() method must be defined '
                              'for subclasses of BaseNode')

  def _set_dependents_dirty(self):
    """Paint all dependent Nodes dirty

    Paints direct dependent Nodes dirty, which causes recursive painting
    for the whole dependent Nodes tree.

    """
    for dependent in self._dependents:
        dependent._set_value(DIRTY)

  def _trigger(self):
    if self._auto_trigger:
      self.get_value()
    for dependent in self._dependents:
      dependent._trigger()

  def _force_inputs_dirty(self):
    self._value = DIRTY

  def _reset_iteration_state(self):
    pass

  def _generate_name(self):
    """Generate a unique name for this Node or Input object

    The name includes:

    * the name of the class
    * an auto-incremented number

    """
    counters = self._name_counters
    template = '{class_name}-{counter}'
    counters[self.__class__] += 1
    return template.format(class_name=self.__class__.__name__,
                           counter=counters[self.__class__])

  def __repr__(self):
    return ('<{self.__class__.__name__} {self.name}: {self._value}>'
            .format(self=self))


class Input(BaseNode):
  """The input node class for reactive programming

  Constructor arguments
  ---------------------

  name (optional): string
          The internal name of the Input. Used in the
          ``__repr__`` of the object. If omitted, a name is
          automatically generated.

  value (optional):
          The initial value for the Input.

  Examples of Inputs::

      >>> input_1 = Input()  # no name, no default value
      >>> input_2 = Input(value=10.0)  # input with a default value
      >>> exponent = Input(name='exponent')  # named input
      >>> sensor = Input(name='sensor', value=-5.3)  # named, with default

  """
  def __init__(self, name):
    super(Input, self).__init__(name)

  def get_value(self):
    return self._value

  def set_value(self, new_value):
    """Set a new value for an Input

    If this caused the value to change, paints dependent Nodes dirty and
    returns the set of those dependent Nodes which are marked "triggered"
    and should be re-evaluated.

    """
    return self._set_value(new_value)

  value = property(get_value, set_value)


@total_ordering
class Node(BaseNode):
  """The Node class for reactive programming

  Constructor arguments
  ---------------------

  name (optional): string
          The internal name of the Node. Used in the ``__repr__`` of the
          object. If omitted, a name is automatically generated.

  action: callable(*positional_inputs, **keyword_inputs)
          The function for calculating the value of a calculated node.
          Values from inputs are provided in positional and keyword arguments
          as defined in the ``inputs=`` argument.

  inputs (optional): ((Input/Node, ...), {key: Input/Node, ...})
          The Nodes and Inputs whose values are used as inputs for the
          action.  This argument can be created with ``Node.inputs()`` which
          provides a cleaner syntax.

  triggered: boolean (default=False)
          ``True`` is this Node shoud be automatically evaluated when any of
          its dependency Nodes or Inputs change value

  Examples of Nodes::

      >>> input_1, input_2, exponent = [Input() for i in range(3)]
      >>> # sum Node with two positional inputs
      >>> sum_node = Node(action=lambda *args: sum(args),
      ...                 inputs=Node.inputs(input_1, input_2))
      >>> # triggered (auto-calculated) Node with two keyword inputs
      >>> triggered_node = Node(
      ...     action=lambda a, x: a ** x,
      ...     inputs=Node.inputs(a=input_1, x=exponent))

  """
  def __init__(self,
               name=None,
               action=None,
               inputs=((), None), auto_trigger=False):
    self._action = action  # must be set before generating name
    super(Node, self).__init__(name, value=DIRTY, auto_trigger=auto_trigger)
    self._positional_inputs = ()
    self._keyword_inputs = {}
    self.set_inputs(*inputs[0], **inputs[1] or {})
    self._set_dependents_dirty()
    self._iteration_reset_state = False

  def _evaluate(self):
    """Calculate the value for the Node

    Calls the action of the Node using values from the inputs of the Node.
    Returns the result of the action function.

    This function can also be overridden in subclasses if a class-based
    approach to creating Node actions is preferred.

    """
    if not self._action:
      raise NotImplementedError('You must define the action= argument '
                                  'when instantiating the Node')
    try:
      positional_values = [i.get_value()
                           for i in self._positional_inputs]
      keyword_values = {name: i.get_value()
                        for name, i in items(self._keyword_inputs)}

      # clear reset state
      self._iteration_reset_state = False
    except StopIteration:
      # reset all input
      self._reset_iteration_state()
      raise StopIteration
    except:
      info = sys.exc_info()
      logger.error('%s:%s'%(info[0], info[1]))
      exit(-1)

    value = self._action(*positional_values, **keyword_values)
    # if (getattr(self._action, 'output_type', None) is not None):
    #     # Output type checking has been enabled, and the node's action
    #     # does specify the expected output type. Check that the calculated
    #     # value matches that type.
    #     self._verify_output_type(value)
    return value

  @staticmethod
  def inputs(*args, **kwargs):
    """Construct a value for the inputs= kwarg of the constructor

    Allows writing this::

        >>> inputs = [Input() for i in range(4)]
        >>> node = Node(inputs=Node.inputs(inputs[0], inputs[1],
        ...                                kw1=inputs[2], kw2=inputs[3]))

    instead of this::

        >>> node = Node(inputs=([inputs[0], inputs[1]],
        ...                     {'kw1': inputs[2], 'kw2': inputs[3]}))

    """
    return args, kwargs

  def output(self, index=0):
    return [self], {}

  def _verify_output_type(self, value):
    """Assert that the given value matches the action's output type

    This check should be run only in development if the developer wants to
    ensure the consistency of a graph's types.

    This method may only be called if the node's action has a non-None
    ``output_type`` attribute.

    Arguments
    ---------
    value: The value whose type is to be checked

    Raises
    ------
    TypeError: The value doesn't match the desired output type of the
               node's action

    """
    if not isinstance(value, self._action.output_type):
      raise TypeError(
          "The output value type {value_type!r} for [{self.name}]\n"
          "doesn't match the expected type "
          '{self._action.output_type.__name__!r} for action '
          '"{self._action.name}".'
          .format(value_type=type(value).__name__, self=self))

  def set_inputs(self, *args, **kwargs):
    """Replace current positional and keyword inputs"""
    for inp in self._iterate_inputs():
        inp._disconnect(self)
    self._positional_inputs = args
    self._keyword_inputs = kwargs
    for inp in self._iterate_inputs():
        inp._connect(self)

  def get_value(self):
    """Return Node value, evaluate if needed and paint dependents dirty"""
    if DIRTY == self._value:
      self._value = self._evaluate()
      #self._set_dependents_dirty()

    return self._value

  def is_dirty(self):
    return DIRTY == self._value

  def iterator_value(self):
    while True:
      self._value = self._evaluate()
      yield self._value
      self._force_inputs_dirty()

  def set_value(self, new_value):
    """Set a new value for an Input

    If this caused the value to change, paints dependent Nodes dirty and
    returns the set of those dependent Nodes which are marked "triggered"
    and should be re-evaluated.

    """
    return self._set_value(new_value)

  value = property(get_value, set_value)

  def _iterate_inputs(self):
    """Iterate through positional and keyword inputs"""
    return itertools.chain(self._positional_inputs,
                           values(self._keyword_inputs))

  def _generate_name(self):
    """Generate a unique name for this Node object

    The name includes:

    * the name of the node class
    * the function name of the ``_action`` if it's defined
      and isn't a lambda
    * an auto-incremented number

    """
    action_name = get_func_name(self._action, '<lambda>')
    if action_name == '<lambda>':
      return super(Node, self)._generate_name()
    counters = self._name_counters
    counters[self.__class__, action_name] += 1
    template = '{class_name}-{action_name}-{counter}'
    return template.format(class_name=self.__class__.__name__,
                           action_name=action_name,
                           counter=counters[self.__class__, action_name])

  def _force_inputs_dirty(self):
    if DIRTY == self._value:
      return
    
    for i in self._positional_inputs:
      i._force_inputs_dirty()
    for name, i in items(self._keyword_inputs):
      i._force_inputs_dirty()
    self._value = DIRTY

  def _reset_iteration_state(self):
    if self._iteration_reset_state:
        return
    for i in self._positional_inputs:
        i._reset_iteration_state()
    for name, i in items(self._keyword_inputs):
        i._reset_iteration_state()
    self._iteration_reset_state = True

  def __lt__(self, other):
    return self.name < other.name


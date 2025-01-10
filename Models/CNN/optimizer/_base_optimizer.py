"""
Base  Optimizer.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

def hello_do_you_copy():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print("Roger that from _base_optimizer.py!")

class _BaseOptimizer:
  def __init__(self, model, learning_rate=1e-4, reg=1e-3):
      self.learning_rate = learning_rate
      self.reg = reg


  def update(self, model):
      pass

  def apply_regularization(self, model):
      '''
      Apply L2 penalty to the model. Update the gradient dictionary in the model
      :param model: The model with gradients
      :return: None, but the gradient dictionary of the model should be updated
      '''

      for m in model.modules:
          if hasattr(m, 'weight'):
              m.dw += self.reg * m.weight


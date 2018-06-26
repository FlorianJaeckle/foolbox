import pytest
import numpy as np

from foolbox.attacks import LinfinityBasicIterativeAttack
from foolbox.attacks import L1BasicIterativeAttack
from foolbox.attacks import L2BasicIterativeAttack
from foolbox.attacks import ProjectedGradientDescentAttack
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
from foolbox.attacks import MomentumIterativeAttack

Attacks = [
    LinfinityBasicIterativeAttack,
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
    ProjectedGradientDescentAttack,
    RandomStartProjectedGradientDescentAttack,
    MomentumIterativeAttack,
]


def test_attack_no_binary_search(bn_adversarial_linf):
    adv = bn_adversarial_linf
    attack = LinfinityBasicIterativeAttack()
    attack(adv, binary_search=False)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_linf(Attack, bn_adversarial_linf):
    adv = bn_adversarial_linf
    attack = Attack()
    attack(adv, binary_search=10)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_l2(Attack, bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_l1(Attack, bn_adversarial_mae):
    adv = bn_adversarial_mae
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_targeted_attack(Attack, bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_gl(Attack, gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is None
    assert adv.distance.value == np.inf

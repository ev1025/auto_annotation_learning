# -*- coding: utf-8 -*-
"""Plan C — 분류 손실을 EQLv2(Equalization Loss v2)로 바꾼다.

무슨 문제를 푸는가
  36클래스 학습에서 한 부품이 배치에 등장할 확률은 몇 %뿐이다. 나머지 프레임 내내 BCE 는
  그 클래스 로짓에 "여긴 배경이다"라는 음(negative) 신호만 보낸다. 누적된 억제가 양(positive)
  신호를 압도하면 그 클래스의 출력이 눌려 아예 발화하지 않는다(진단 결과 test4 의 Miss 지배).

EQLv2 (Tan et al., CVPR 2021)
  클래스별로 누적 양·음 그래디언트를 세고 그 비율 g_j = pos_j / neg_j 로 가중치를 정한다.
      r_j     = sigmoid(gamma * (g_j - mu))        비율이 낮은(눌린) 클래스일수록 0 에 가깝다
      pos_w_j = 1 + alpha * (1 - r_j)              눌린 클래스의 양 신호를 키우고
      neg_w_j = r_j                                눌린 클래스의 음 신호를 줄인다
  손실 = BCE * (target * pos_w + (1-target) * neg_w)

ultralytics 적응
  v8DetectionLoss 는 분류 손실을 `self.bce(pred_scores, target_scores)` 한 줄로 부르고
  요소별(bs, anchors, nc) 값을 돌려받아 합산한다. 그래서 self.bce 를 이 모듈로 바꿔 끼우면
  나머지 학습 경로(TAL 할당·박스 손실·DFL)는 그대로 둔 채 분류 손실만 교체된다.
  target_scores 는 0/1 이 아니라 TAL 정렬 점수(soft, 0~1)라 target 을 그대로 양 비중으로 쓴다.

  venv 의 ultralytics 파일은 건드리지 않는다. install() 이 클래스 __init__ 만 감싼다.

사용: run_exp.py --eqlv2
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 12.0     # 비율 -> 가중치 매핑의 기울기 (논문 기본값)
MU = 0.8         # 매핑의 중심 (논문 기본값)
ALPHA = 4.0      # 양 신호를 최대 몇 배까지 키울지 (논문 기본값)


class EQLv2(nn.Module):
    """BCEWithLogitsLoss(reduction='none') 자리에 그대로 들어가는 대체품."""

    def __init__(self, nc: int, gamma: float = GAMMA, mu: float = MU, alpha: float = ALPHA):
        super().__init__()
        self.nc, self.gamma, self.mu, self.alpha = nc, gamma, mu, alpha
        # 누적량은 학습 전체에 걸쳐 쌓인다. 버퍼로 두면 device 이동·체크포인트가 따라온다.
        self.register_buffer("pos_grad", torch.zeros(nc))
        self.register_buffer("neg_grad", torch.zeros(nc))
        self.register_buffer("pos_neg", torch.full((nc,), 100.0))   # 초기엔 억제 없음(r≈1)

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gamma * (x - self.mu))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: 로짓 (bs, anchors, nc) · target: TAL 정렬 점수 (같은 모양, 0~1)."""
        r = self._map(self.pos_neg).to(pred.dtype)                  # (nc,)
        pos_w = (1.0 + self.alpha * (1.0 - r)).view(1, 1, -1)
        neg_w = r.view(1, 1, -1)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        weight = target * pos_w + (1.0 - target) * neg_w
        out = loss * weight

        with torch.no_grad():        # 다음 스텝의 가중치를 정하려고 이번 스텝 그래디언트를 센다
            prob = pred.detach().float().sigmoid()
            t = target.detach().float()
            grad = (t * (1.0 - prob) + (1.0 - t) * prob) * weight.detach().float()   # |dL/d로짓|
            self.pos_grad += (grad * t).sum(dim=(0, 1))
            self.neg_grad += (grad * (1.0 - t)).sum(dim=(0, 1))
            self.pos_neg.copy_(self.pos_grad / (self.neg_grad + 1e-10))
        return out

    def report(self, names=None, k: int = 6):
        """가장 억제된 클래스 k 개(비율이 낮을수록 눌려 있다)."""
        pn = self.pos_neg.detach().cpu()
        idx = torch.argsort(pn)[:k].tolist()
        f = (lambda i: names[i]) if names else str
        return {f(i): round(float(pn[i]), 5) for i in idx}


_INSTALLED = False


def install(gamma: float = GAMMA, mu: float = MU, alpha: float = ALPHA):
    """v8DetectionLoss 가 만들어질 때 self.bce 를 EQLv2 로 바꿔 끼운다(전역 1회)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import ultralytics.utils.loss as L

    orig = L.v8DetectionLoss.__init__

    def patched(self, model, *a, **k):
        orig(self, model, *a, **k)
        if getattr(self, "class_weights", None) is not None:
            # class_weights 와 EQLv2 를 같이 쓰면 재가중이 두 번 걸린다
            raise RuntimeError("EQLv2 와 class_weights 를 동시에 쓸 수 없습니다")
        self.bce = EQLv2(self.nc, gamma, mu, alpha).to(self.device)

    L.v8DetectionLoss.__init__ = patched
    _INSTALLED = True

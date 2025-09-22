"""
自定义Cornac兼容的Coverage评估指标 - 修复版本
"""
import numpy as np
from cornac.metrics import RankingMetric

class Coverage(RankingMetric):
    def __init__(self, k=10, catalog_size=None):
        super().__init__(k=k)
        self.name = f"Coverage@{k}"
        self.catalog_size = catalog_size
        self.recommended_items = set()
        print(f"Coverage metric initialized with k={k}, catalog_size={catalog_size}")

    def reset(self):
        self.recommended_items = set()
        print("Coverage metric reset")

    def compute(self, gt_pos, pd_rank, **kwargs):
        if self.catalog_size is None:
            raise ValueError("Catalog size must be set before evaluation")

        if pd_rank is not None and len(pd_rank) > 0:
            # 调试：检查推荐物品的类型和值
            if len(self.recommended_items) < 10:  # 只在前几次显示调试信息
                print(f"Adding recommendations: {pd_rank[:min(5, len(pd_rank))]}...")

            # 确保物品ID是整数
            try:
                valid_items = []
                for item in pd_rank[:self.k]:
                    try:
                        valid_items.append(int(item))
                    except (ValueError, TypeError):
                        # 如果无法转换为整数，尝试直接使用
                        valid_items.append(item)

                self.recommended_items.update(valid_items)

                # 调试信息
                if len(self.recommended_items) < 10:
                    print(f"Current unique items: {len(self.recommended_items)}")

            except Exception as e:
                print(f"Error processing recommendations: {e}")
                # 备选方案：直接添加
                self.recommended_items.update(pd_rank[:self.k])

        return 0.0

    def value(self):
        if self.catalog_size is None or self.catalog_size == 0:
            return 0.0

        coverage_value = len(self.recommended_items) / self.catalog_size
        print(f"Coverage calculation: {len(self.recommended_items)} / {self.catalog_size} = {coverage_value:.4f}")
        return coverage_value

    def get_recommended_count(self):
        return len(self.recommended_items)
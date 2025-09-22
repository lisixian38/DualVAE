import pandas as pd
import argparse
import cornac
from cornac.metrics import Recall, NDCG
from models.recom_dualvaecf import DualVAECF
from cornac_metrics import Coverage  # 导入自定义的Coverage指标

parser = argparse.ArgumentParser(description="DualVAE model")  # 超参数配置
parser.add_argument("-d", "--dataset", type=str, default="ML1M",  # default默认
                    help="name of the dataset, suppose ['ML1M', 'AKindle', 'Yelp']")  # 数据集名
parser.add_argument("-k", "--latent_dim", type=int, default=20,
                    help="number of the latent dimensions")  # 潜在维度
parser.add_argument("-a", "--num_disentangle", type=int, default=5,
                    help="number of the disentangled representation")  # 解耦表示数量
parser.add_argument("-en", "--encoder", type=str, default="[40]",
                    help="structure of the user/item encoders")  # 编码器
parser.add_argument("-de", "--decoder", type=str, default="[40]",
                    help="structure of the user/item decoder")  # 解码器
parser.add_argument("-af", "--act_fn", type=str, default="tanh",
                    choices=["sigmoid", "tanh", "relu", "relu6", "elu"],
                    help="non-linear activation function for the encoders")  # 激活函数
parser.add_argument("-lh", "--likelihood", type=str, default="pois",
                    choices=["pois", "bern", "gaus", "mult"],
                    help="likelihood function to fit the rating observations")
parser.add_argument("-ne", "--num_epochs", type=int, default=200,
                    help="number of training epochs")  # 训练轮数
parser.add_argument("-bs", "--batch_size", type=int, default=128,
                    help="batch size for training")  # 批量大小
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="learning rate for training")  # 学习率
parser.add_argument("-kl", "--beta_kl", type=float, default=1.0,
                    help="beta weighting for the KL divergence")  # KL散度权重β
parser.add_argument("-cl", "--gama_cl", type=float, default=0.01,
                    help="gama weighting for the contrast loss")  # 对比学习损失的权重γ
parser.add_argument("-tn", "--top_n", type=int, default=[20, 50],
                    help="n cut-off for top-n evaluation")  # 评估指标
parser.add_argument("-s", "--random_seed", type=int, default=123,
                    help="random seed value")  # 设置随机种子，确保每次运行结果一致
parser.add_argument("-v", "--verbose", default=True,
                    help="increase output verbosity")  # 控制是否在运行过程中显示详细信息
parser.add_argument("-gpu", "--gpu", type=int, default=0,
                    help="gpu-id")
parser.add_argument("--evaluate", action="store_true",
                    help="Run evaluation after training")
args = parser.parse_args()
print(args)


def gen_cornac_dataset(data_path, t=-1):
    df = pd.read_csv(data_path)
    if df.shape[1] == 2:
        df.insert(loc=2, column='rating', value=5.0)
    df = df[df.rating > t]
    return df


def load_dataset():
    if args.dataset in ['100K', '1M', '10M']:
        data = cornac.datasets.movielens.load_feedback(variant=args.dataset)
        eval_method = cornac.eval_methods.RatioSplit(data=data, test_size=0.2, rating_threshold=1.0, seed=123,
                                                     verbose=args.verbose)
    else:
        dataset_dir = f"./data/{args.dataset}/"
        train_data = gen_cornac_dataset(dataset_dir + 'train.csv')
        test_data = gen_cornac_dataset(dataset_dir + 'test.csv')
        eval_method = cornac.eval_methods.BaseMethod.from_splits(
            train_data=train_data.values,
            test_data=test_data.values,
            seed=args.random_seed,
            verbose=args.verbose,
            rating_threshold=1.0,
        )
    return eval_method


if __name__ == "__main__":
    eval_method = load_dataset()

    dualvae = DualVAECF(
        k=args.latent_dim,
        a=args.num_disentangle,
        encoder_structure=eval(args.encoder),
        decoder_structure=eval(args.decoder),
        act_fn=args.act_fn,
        likelihood=args.likelihood,
        n_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_kl=args.beta_kl,
        gama_cl=args.gama_cl,
        seed=args.random_seed,
        gpu=args.gpu,
        verbose=args.verbose,
    )

    topk_metrics = [Recall(args.top_n), NDCG(args.top_n)]

    experiment = cornac.Experiment(
        eval_method=eval_method,
        models=[dualvae],
        metrics=topk_metrics,
        user_based=True
    )

    experiment.run()

    if args.evaluate:
        print("\n" + "=" * 60)
        print("Additional Coverage Metrics")
        print("=" * 60)

        try:
            catalog_size = eval_method.train_set.matrix.shape[1]
            print(f"Catalog size: {catalog_size}")

            user_ids = list(eval_method.test_set.user_ids)
            print(f"Number of test users: {len(user_ids)}")

            if isinstance(args.top_n, list):
                top_n_list = args.top_n
            else:
                top_n_list = [args.top_n]

            for top_n in top_n_list:
                print(f"\nComputing Coverage@{top_n}...")
                coverage_metric = Coverage(k=top_n, catalog_size=catalog_size)
                coverage_metric.reset()

                successful_users = 0

                for i, user_id in enumerate(user_ids):
                    try:
                        # 尝试多种方式获取推荐
                        scores = None

                        # 方式1：使用用户ID字符串
                        try:
                            scores = dualvae.score(str(user_id))
                        except:
                            pass

                        # 方式2：使用用户ID数值
                        if scores is None or len(scores) == 0:
                            try:
                                scores = dualvae.score(user_id)
                            except:
                                pass

                        # 方式3：使用用户索引
                        if scores is None or len(scores) == 0:
                            try:
                                user_idx = i  # 使用循环索引
                                scores = dualvae.score(user_idx)
                            except:
                                pass

                        if scores is None or len(scores) == 0:
                            continue

                        # 获取推荐物品
                        recommended = scores.argsort()[-top_n:][::-1]

                        # 确保物品ID在有效范围内
                        valid_recommended = []
                        for item in recommended:
                            if 0 <= item < catalog_size:
                                valid_recommended.append(item)

                        if len(valid_recommended) > 0:
                            coverage_metric.compute(None, valid_recommended)
                            successful_users += 1

                            # 显示前几个用户的调试信息
                            if successful_users <= 3:
                                print(f"User {user_id}: recommended {len(valid_recommended)} items")

                    except Exception as e:
                        continue

                coverage_value = coverage_metric.value()
                print(f"Coverage@{top_n}: {coverage_value:.4f}")
                print(f"  Unique items recommended: {coverage_metric.get_recommended_count()}")
                print(f"  Total catalog size: {catalog_size}")
                print(f"  Coverage percentage: {coverage_value * 100:.2f}%")
                print(f"  Successful users: {successful_users}/{len(user_ids)}")
                print("-" * 40)

        except Exception as e:
            print(f"Error computing coverage: {e}")
            import traceback

            traceback.print_exc()
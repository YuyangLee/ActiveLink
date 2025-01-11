import openke
import os
import argparse
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR, ComplEx, RotatE
from openke.module.loss import MarginLoss, SigmoidLoss, SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Train and test a knowledge graph embedding model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["TransE", "TransR", "RotatE", "ComplEx"],
        required=True,
        help="The model to train (TransE, TransR, RotatE, ComplEx).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing the dataset.",
    )
    args = parser.parse_args()

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=args.data_dir,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # dataloader for testing
    test_dataloader = TestDataLoader(args.data_dir, "link", type_constrain=False)

    # select the model and loss function based on the argument
    if args.model == "TransE":
        model_instance = TransE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=200,
            p_norm=1,
            norm_flag=True,
        )
        loss_function = NegativeSampling(
            model=model_instance,
            loss=MarginLoss(margin=5.0),
            batch_size=train_dataloader.get_batch_size(),
        )
    elif args.model == "RotatE":
        model_instance = RotatE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=1024,
            margin=6.0,
            epsilon=2.0,
        )
        loss_function = NegativeSampling(
            model=model_instance,
            loss=SigmoidLoss(adv_temperature=2),
            batch_size=train_dataloader.get_batch_size(),
            regul_rate=0.0,
        )
    elif args.model == "TransR":
        model_instance = TransR(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim_e=200,
            dim_r=200,
            p_norm=1,
            norm_flag=True,
            rand_init=False,
        )
        loss_function = NegativeSampling(
            model=model_instance,
            loss=MarginLoss(margin=4.0),
            batch_size=train_dataloader.get_batch_size(),
        )
    elif args.model == "ComplEx":
        model_instance = ComplEx(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=200,
        )
        loss_function = NegativeSampling(
            model=model_instance,
            loss=SoftplusLoss(),
            batch_size=train_dataloader.get_batch_size(),
            regul_rate=1.0,
        )

    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")

    # train the model
    trainer = Trainer(
        model=loss_function,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=1.0,
        use_gpu=True,
    )
    trainer.run()
    model_checkpoint_path = (
        f"./checkpoint/{args.model.lower()}_{os.path.basename(args.data_dir)}.ckpt"
    )
    model_instance.save_checkpoint(model_checkpoint_path)

    # test the model
    model_instance.load_checkpoint(model_checkpoint_path)
    tester = Tester(model=model_instance, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)


if __name__ == "__main__":
    main()

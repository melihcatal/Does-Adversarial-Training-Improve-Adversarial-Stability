# pylint: skip-file
import os

import advsecurenet.shared.types.configs.attack_configs as AttackConfigs
import click
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import torch
from advsecurenet.attacks import (FGSM, LOTS, PGD, CWAttack, DecisionBoundary,
                                  DeepFool, TargetedFGSM)
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.datasets import DatasetFactory
from advsecurenet.evaluation.evaluators.transferability_evaluator import \
    TransferabilityEvaluator
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types import DatasetType
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from advsecurenet.utils.data import unnormalize_data
from advsecurenet.utils.model_utils import load_model
from torchvision import transforms
from tqdm.auto import tqdm


def get_attacks(device):
    # Constants
    attacks = {
        # "fgsm": {
        #     "config": AttackConfigs.FgsmAttackConfig(epsilon=0.1, device=device),
        #     "attack": FGSM,
        #     "targeted": False
        # },
        # "fgsm_targeted": {
        #     "config": AttackConfigs.FgsmAttackConfig(epsilon=0.1, device=device),
        #     "attack": TargetedFGSM,
        #     "targeted": True
        # },
        # "pgd_single": {
        #     "config": AttackConfigs.PgdAttackConfig(epsilon=0.1, device=device, num_iter=1),
        #     "attack": PGD,
        #     "targeted": False
        # },
        # "pgd_iterative": {
        #     "config": AttackConfigs.PgdAttackConfig(epsilon=0.1, device=device, num_iter=10),
        #     "attack": PGD,
        #     "targeted": False
        # },

        "deepfool_single": {
            "config": AttackConfigs.DeepFoolAttackConfig(device=device, num_classes=1000, max_iterations=1),
            "attack": DeepFool,
            "targeted": False
        },
        "deepfool_iterative": {
            "config": AttackConfigs.DeepFoolAttackConfig(device=device, num_classes=1000, max_iterations=10),
            "attack": DeepFool,
            "targeted": False
        },
        "lots_single": {
            "config": AttackConfigs.LotsAttackConfig(deep_feature_layer="placeholder",
                                                     mode=AttackConfigs.LotsAttackMode.SINGLE,
                                                     max_iterations=1000,
                                                     learning_rate=0.1,
                                                     epsilon=0.1,
                                                     device=device),
            "attack": LOTS,
            "targeted": True
        },
        "lots_iterative": {
            "config": AttackConfigs.LotsAttackConfig(deep_feature_layer="placeholder",
                                                     mode=AttackConfigs.LotsAttackMode.ITERATIVE,
                                                     max_iterations=10,
                                                     learning_rate=0.1,
                                                     epsilon=0.1,
                                                     device=device),
            "attack": LOTS,
            "targeted": True
        }
        # ,
        # "decision_boundary_non_targeted": {
        #     "config": AttackConfigs.DecisionBoundaryAttackConfig(device=device, early_stopping=False, max_iterations=10, targeted=False),
        #     "attack": DecisionBoundary,
        #     "targeted": False
        # },
        # "decision_boundary_targeted": {
        #     "config": AttackConfigs.DecisionBoundaryAttackConfig(device=device, early_stopping=False, max_iterations=10, targeted=True),
        #     "attack": DecisionBoundary,
        #     "targeted": True
        # }
    }
    return attacks


def get_to_skip_combinations():
    to_skip_combinations = {}

    return to_skip_combinations


def load_robust_resnet18(device):
    # load the robust resnet18
    robust_resnet18 = ModelFactory.create_model(
        "resnet18", pretrained=False, num_classes=1000).to(device)
    robust_resnet18 = load_model(
        robust_resnet18, "/home/paperspace/17jan/lots/resnet18.pth"
    )
    return robust_resnet18


def load_models(device):
    models = {
        "resnet18": load_robust_resnet18(device),
        "convnext_tiny": ModelFactory.create_model("convnext_tiny", pretrained=True, num_classes=1000).to(device),
        "convnext_small": ModelFactory.create_model("convnext_small", pretrained=True, num_classes=1000).to(device),
        "resnet152": ModelFactory.create_model("resnet152", pretrained=True, num_classes=1000).to(device),
        "vit_l_32": ModelFactory.create_model("vit_l_32", pretrained=True, num_classes=1000).to(device),
        "vit_b_16": ModelFactory.create_model("vit_b_16", pretrained=True, num_classes=1000).to(device),
        "swin_v2_s": ModelFactory.create_model("swin_v2_s", pretrained=True, num_classes=1000).to(device),
        "swin_v2_t": ModelFactory.create_model("swin_v2_t", pretrained=True, num_classes=1000).to(device),
        "densenet121": ModelFactory.create_model("densenet121", pretrained=True, num_classes=1000).to(device),
        "densenet201": ModelFactory.create_model("densenet201", pretrained=True, num_classes=1000).to(device),
    }
    return models


def load_and_sample_imagenet(
        root_dir="/home/paperspace/imagenet/val",
        num_samples_per_class=5,
        specific_folders=None):
    """
    Load ImageNet data and sample images from specific folders only.

    Args:
    - root_dir: Root directory of the ImageNet dataset.
    - num_samples_per_class: Number of samples per class to include.
    - specific_folders: List of folder names to include (e.g., ['n01534433']).

    Returns:
    - sampled_dataset: A torch dataset of the sampled images.
    """

    # Load the dataset
    dataset = DatasetFactory.create_dataset(DatasetType.IMAGENET)
    test_data = dataset.load_dataset(root=root_dir, train=False)

    # Sample images
    sampled_indices = []
    class_counts = {}
    for idx, (path, class_idx) in tqdm(enumerate(test_data.dataset.imgs),
                                       total=len(test_data),
                                       desc="Sampling Images",
                                       leave=False,
                                       colour="blue"):
        folder_name = os.path.split(os.path.dirname(path))[-1]
        if specific_folders and folder_name not in specific_folders:
            continue  # Skip images not in the specified folders

        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        if class_counts[class_idx] <= num_samples_per_class:
            sampled_indices.append(idx)

    # Create a subset
    sampled_dataset = torch.utils.data.Subset(test_data, sampled_indices)

    return sampled_dataset


def prepare_data():
    subset_dataset = load_and_sample_imagenet()
    test_loader = DataLoaderFactory.create_dataloader(
        dataset=subset_dataset, batch_size=16, shuffle=True, num_workers=2)
    return test_loader


def evaluate_transferability(source_model,
                             target_models,
                             attack,
                             dataloader,
                             mean,
                             std,
                             targeted=False,
                             adversarial_target_generator=AdversarialTargetGenerator()):
    # Evaluate the transferability of the attack
    with TransferabilityEvaluator(target_models) as evaluator:
        # get the device of the source model
        device = next(source_model.parameters()).device
        for images, labels in tqdm(dataloader, desc="Evaluating Transferability", colour="red", leave=False):
            # move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # try:
            # Conditional handling for targeted vs. non-targeted attacks
            if targeted:
                # Generate target images and labels for targeted attacks
                paired = adversarial_target_generator.generate_target_images(
                    zip(images, labels)
                    # , total_tries=10
                )
                original_images, original_labels, target_images, target_labels = adversarial_target_generator.extract_images_and_labels(
                    paired, images, device)

                # Unnormalize the images
                unnormalized_images = unnormalize_data(
                    original_images, mean, std)
                # Unnormalize the target images
                unnormalized_target_images = unnormalize_data(
                    target_images, mean, std)

                if isinstance(attack, LOTS):
                    # Perform LOTS attack
                    adv_images, is_found = attack.attack(
                        model=source_model,
                        data=unnormalized_images,
                        target=unnormalized_target_images,
                        target_classes=target_labels,
                    )
                elif isinstance(attack, DecisionBoundary):
                    adv_images = attack.attack(
                        source_model, unnormalized_images, original_labels, target_labels)

                else:

                    # Perform other types of targeted attacks
                    adv_images = attack.attack(
                        source_model, unnormalized_images, target_labels)

                # Normalize the images since the evaluator expects normalized data
                normalized_images = transforms.Normalize(
                    mean, std)(unnormalized_images)
                normalized_adv_images = transforms.Normalize(
                    mean, std)(adv_images)

                evaluator.update(source_model=source_model, original_images=normalized_images, true_labels=original_labels,
                                 adversarial_images=normalized_adv_images, is_targeted=True, target_labels=target_labels)
            else:

                # unnormalize the images
                unnormalized_images = unnormalize_data(
                    images, mean, std)

                # Perform standard attack for non-targeted attacks
                adv_images = attack.attack(
                    source_model, unnormalized_images, labels)

                # Normalize the images since the evaluator expects normalized data
                normalized_images = transforms.Normalize(mean, std)(images)
                normalized_adv_images = transforms.Normalize(
                    mean, std)(adv_images)

                # Evaluate the transferability of the attack
                evaluator.update(source_model, normalized_images,
                                 labels, normalized_adv_images)
            # except Exception as e:
            #     click.secho(
            #         f"Error occurred while evaluating transferability for {attack} on {source_model}", fg="red")
            #     click.secho(str(e), fg="red")
            #     continue

            # free up memory
            images.to("cpu")
            labels.to("cpu")
            adv_images.to("cpu")

        transferability_results = evaluator.get_results()
        return transferability_results, evaluator.transferability_data, evaluator.total_successful_on_source


def save_results(dataframe,
                 path,
                 raw_transferability=None,
                 raw_successful_on_source=None,
                 filename="transferability.csv"):
    save_path = os.path.join(path, filename)
    # if exists, append to the file
    if os.path.exists(save_path):
        current_dataframe = pd.read_csv(save_path, index_col=0)
        # merge the dataframes
        dataframe = pd.concat([current_dataframe, dataframe], axis=1)

    # save the dataframe
    dataframe.to_csv(save_path)

    # if given, save the raw data to a log txt file
    if raw_transferability is not None and raw_successful_on_source is not None:
        log_path = os.path.join(path, "log.txt")
        with open(log_path, "w") as f:
            f.write(f"Transferability Data: {raw_transferability}\n")
            f.write(f"Successful on Source: {raw_successful_on_source}\n")


def prepare_attack(source_model, attack):
    if attack["attack"] is LOTS:
        attack["config"].deep_feature_layer = f"model.{
            source_model.get_layer_names()[-1]}"
    return attack["attack"](attack["config"])


def freeup_memory():
    torch.cuda.empty_cache()


@click.command()
@click.option("--device", "-d", default="cuda", help="Device to use for training.")
@click.option("--attack_to_use", "-a", default="all", help="Attack to use for transferability evaluation.")
@click.option("--target_model", "-t", default="all", help="Target model to use for transferability evaluation.")
def main(device, attack_to_use, target_model):

    adversarial_target_generator = AdversarialTargetGenerator()

    attacks = get_attacks(device)

    if attack_to_use != "all":
        # filter the attacks
        attacks = {key: value for key,
                   value in attacks.items() if key == attack_to_use}

    to_skip_combinations = get_to_skip_combinations()

    # Main Execution
    click.echo(click.style("Starting Transferability Evaluation.", fg="green"))
    models = load_models(device)
    dataloader = prepare_data()

    # mean and std for unnormalizing the data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for attack_name, attack in tqdm(attacks.items(), desc=f"Cross Evaluating Transferability", total=len(attacks), leave=False, position=0, dynamic_ncols=True, colour="yellow"):

        transferability_results = {}
        completed_models = []

        for source_model_name, source_model in tqdm(models.items(), desc=f"Model", total=len(models), leave=False, position=1, dynamic_ncols=True, colour="yellow"):

            target_models = list(models.values())
            target_models = [
                model for model in target_models if model.model_name == target_model]

            # remove the source model from the target models
            # target_models = [
            #     model for model in target_models if model.model_name != source_model_name]

            # if source and target model are the same, set results to 1
            if source_model_name == target_model:
                transferability_results[source_model_name] = 1
                continue

            # remove the to "skip target models"
            try:
                for combination in to_skip_combinations[attack_name]:
                    if combination["source_model"] == source_model_name:
                        target_models = [
                            model for model in target_models if model.model_name not in combination["target_models"]]
                        break
            except KeyError:
                pass

            if target_models is None or len(target_models) == 0:
                continue

            if source_model_name in completed_models:
                continue
            # click.echo(
            #     f"Source Model: {source_model_name} Attack: {attack_name} Target Models: {[model.model_name for model in target_models]}")

            device = next(source_model.parameters()).device

            targeted = attack["targeted"]
            attack_obj = prepare_attack(source_model, attack)

            results, raw_transferability, raw_successful_on_source = evaluate_transferability(
                source_model,
                target_models,
                attack_obj,
                dataloader,
                targeted=targeted,
                adversarial_target_generator=adversarial_target_generator,
                mean=mean,
                std=std
            )

            transferability_results[source_model_name] = results

            # save temporary results
            dataframe = pd.DataFrame(transferability_results)

            save_path = f"temp/{attack_name}/{source_model_name}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save temporary results
            save_results(dataframe, save_path, raw_transferability=raw_transferability,
                         raw_successful_on_source=raw_successful_on_source)
            # free up memory
            freeup_memory()

        if len(transferability_results) == 0:
            continue
        # create a dataframe from the results
        dataframe = pd.DataFrame(transferability_results)

        # save the final results
        save_path = f"results/{attack_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_results(dataframe, save_path)

    click.echo(click.style("Done!", fg="green"))


if __name__ == "__main__":
    main()

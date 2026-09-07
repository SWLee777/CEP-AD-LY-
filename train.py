import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from CEP_AD import CEP
from loss import FocalLoss, BinaryDiceLoss, BinaryFocalLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import os
import random
from utils import get_transform, compute_gradient_foreground

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = CEP(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)

    for n, p in prompt_learner.named_parameters():
        if p.requires_grad == True:
            print(n)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_fun = BinaryFocalLoss()
    loss_mse = torch.nn.MSELoss()

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []
        image_loss_list2 = []
        loss_etf_text_list = []
        vis_cls_loss_list = []
        loss_etf_sca_list = []
        loss_balance_list = []

        for items in tqdm(train_dataloader):

            image = items['img'].to(device)
            label =  items['anomaly']
            class_name = items['cls_name']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts_pos, prompts_neg, tokenized_prompt_pos, tokenized_prompt_neg, compound_prompts_text, _ = prompt_learner.forward()
            text_features_pos = model.encode_text_learn(prompts_pos, tokenized_prompt_pos, compound_prompts_text).float()
            text_features_neg = model.encode_text_learn(prompts_neg, tokenized_prompt_neg, compound_prompts_text).float()

            # ETF loss on text negative prompts: push them toward equiangular tight frame
            n_neg = text_features_neg.shape[0]
            neg_norm = F.normalize(text_features_neg, dim=-1)
            gram_text = neg_norm @ neg_norm.T
            target_text = torch.full((n_neg, n_neg), -1.0 / (n_neg - 1), device=device, dtype=gram_text.dtype)
            target_text.fill_diagonal_(1.0)
            loss_etf_text = F.mse_loss(gram_text, target_text)
            loss_etf_text_list.append(loss_etf_text.item())

            text_features_neg = torch.mean(text_features_neg, dim=0, keepdim=True)
            text_features = torch.cat([text_features_pos, text_features_neg])
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # [1370, B, 768]
            similarity, _ = AnomalyCLIP_lib.compute_similarity_ori(patch_features, text_features[0])
            similarity_map1 = AnomalyCLIP_lib.get_similarity_map(similarity[1:, :], args.image_size)
            map_max_score1 = similarity[1:, :, 1].max(dim=0).values

            fg_score = compute_gradient_foreground(image, patch_size=14)  # [1369, B]

            text_features_list = []
            vis_score_list = []
            loss_bias = 0.0
            gate_probs_list = []
            expert_outputs_list = []
            for i in range(image.shape[0]):
                patch_feat_i = patch_features[1:, i, :]  # [1369, 768]
                fg_score_i   = fg_score[:, i]            # [1369]

                prompts_pos, prompts_neg, tokenized_prompt_pos, tokenized_prompt_neg, compound_prompts_text, bias = \
                    prompt_learner.forward(patch_features=patch_feat_i, fg_score=fg_score_i, fg_beta=args.fg_beta)
                text_features_pos = model.encode_text_learn(prompts_pos, tokenized_prompt_pos, compound_prompts_text).float()
                text_features_neg = model.encode_text_learn(prompts_neg, tokenized_prompt_neg, compound_prompts_text).float()
                text_features_neg = torch.mean(text_features_neg, dim=0, keepdim=True)

                text_features = torch.cat([text_features_pos, text_features_neg])
                text_features_list.append(text_features)

                # Collect MoE routing info for ETF-SCA and load-balance losses
                if prompt_learner._gate_probs is not None:
                    gate_probs_list.append(prompt_learner._gate_probs)
                    expert_outputs_list.append(prompt_learner._expert_outputs)

                bias_gt0 = torch.zeros(768).to(device)
                if label[i] == 0:
                    loss_tmp = loss_mse(bias.squeeze(0), bias_gt0)
                    loss_bias = loss_bias + loss_tmp

                vis_score_i = torch.sigmoid(prompt_learner.vis_cls_head(bias))  # [1, 1]
                vis_score_list.append(vis_score_i)

            vis_scores = torch.cat(vis_score_list).squeeze(-1)  # [B]
            loss_vis_cls = F.binary_cross_entropy(vis_scores, label.float().to(device))
            vis_cls_loss_list.append(loss_vis_cls.item())

            # ETF loss on visual MoE expert outputs (forces experts to be diverse/orthogonal)
            # Load-balance loss: penalizes routing collapse (CV² of gate sums across batch)
            loss_etf_sca = torch.tensor(0.0, device=device)
            loss_balance = torch.tensor(0.0, device=device)
            if expert_outputs_list:
                n_exp = expert_outputs_list[0].shape[0]
                target_exp = torch.full((n_exp, n_exp), -1.0 / (n_exp - 1), device=device, dtype=torch.float32)
                target_exp.fill_diagonal_(1.0)
                for exp_out in expert_outputs_list:
                    exp_norm = F.normalize(exp_out.float(), dim=-1)
                    loss_etf_sca = loss_etf_sca + F.mse_loss(exp_norm @ exp_norm.T, target_exp)
                loss_etf_sca = loss_etf_sca / len(expert_outputs_list)
                gate_batch = torch.stack(gate_probs_list)  # [B, num_experts]
                gate_sum = gate_batch.sum(dim=0)
                loss_balance = (gate_sum.std() / (gate_sum.mean() + 1e-6)).pow(2)
            loss_etf_sca_list.append(loss_etf_sca.item())
            loss_balance_list.append(loss_balance.item())

            text_features = torch.stack(text_features_list)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_features, text_features)
            similarity_map2 = AnomalyCLIP_lib.get_similarity_map(similarity[1:, :], args.image_size)
            map_max_score2 = similarity[1:, :, 1].max(dim=0).values

            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())

            tmp = (text_probs).softmax(-1)
            tmp = tmp[:, 1]

            map_max_score = (2 * map_max_score1 + map_max_score2)/3
            score2 = 0.5 * (tmp + map_max_score)
            image_loss2 = loss_fun(score2, label.float().to(device))
            image_loss_list2.append(image_loss2.item())

            loss = 0
            loss += loss_focal(similarity_map1, gt)
            loss += loss_dice(similarity_map1[:, 1, :, :], gt)
            loss += loss_dice(similarity_map1[:, 0, :, :], 1-gt)

            loss += 0.5 * loss_focal(similarity_map2, gt)
            loss += 0.5 * loss_dice(similarity_map2[:, 1, :, :], gt)
            loss += 0.5 * loss_dice(similarity_map2[:, 0, :, :], 1 - gt)

            optimizer.zero_grad()
            # Contrastive loss across all 4 scale branches: pull each branch's
            # anomaly/normal token pair apart (cos sim < -0.5).
            cos_sim_tokens = 0.0
            n_branch = len(prompt_learner.vis_token_adapter.anomaly_vis_tokens)
            for bi in range(n_branch):
                a_tok = prompt_learner.vis_token_adapter.anomaly_vis_tokens[bi]
                n_tok = prompt_learner.vis_token_adapter.normal_vis_tokens[bi]
                cs = F.cosine_similarity(a_tok.unsqueeze(0), n_tok.unsqueeze(0), dim=1)
                cos_sim_tokens = cos_sim_tokens + cs
            cos_sim_tokens = cos_sim_tokens / n_branch
            loss_contrastive = torch.clamp(cos_sim_tokens + 0.5, min=0).mean()
            (loss + image_loss + image_loss2 + 0.1*loss_etf_text + loss_bias + 0.1*loss_contrastive + loss_vis_cls + 0.1*loss_etf_sca + 0.01*loss_balance).backward()
            optimizer.step()
            loss_list.append(loss.item())


        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, image_loss2:{:.4f}, etf_text:{:.4f}, vis_cls:{:.4f}, etf_sca:{:.4f}, balance:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list), np.mean(image_loss_list2), np.mean(loss_etf_text_list), np.mean(vis_cls_loss_list), np.mean(loss_etf_sca_list), np.mean(loss_balance_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--fg_beta", type=float, default=0.5, help="gradient foreground weight for top-k selection")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)

import torch
import torch.nn as nn
import timm
import torch.optim as optim
import torch.nn.init as init


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classification_parallel(nn.Module):
    def __init__(self, model_name, device):
        super(Classification_parallel, self).__init__()
        self.model_name = model_name
        self.model_age = timm.create_model(self.model_name, pretrained=True, num_classes=3).to(device=device)
        self.model_gender = timm.create_model(self.model_name, pretrained=True, num_classes=2).to(device=device)
        self.model_mask = timm.create_model(self.model_name, pretrained=True, num_classes=3).to(device=device)


        class_num_final = [2730, 2005, 375, 3595, 4050, 505, 546, 401, 75, 719, 810, 101, 943, 885 ,276, 1747 ,1125, 181]
        class_num_mask = [13260,2652,5157]
        class_num_gender = [8236, 12833]
        class_num_age = [10280,9276,1513]
        for i in range(len(class_num_final)):
            class_num_final[i] = 18900/(18 * class_num_final[i])
        for i in range(len(class_num_mask)):
            class_num_mask[i] = 18900/(3 * class_num_mask[i])
        for i in range(len(class_num_gender)):
            class_num_gender[i] = 18900/(2 * class_num_gender[i])
        for i in range(len(class_num_age)):
            class_num_age[i] = 18900/(3 * class_num_age[i])
        class_weight_final = torch.tensor(class_num_final).to(device=device, dtype=torch.float)
        class_weight_mask = torch.tensor(class_num_mask).to(device=device, dtype=torch.float)
        class_weight_gender = torch.tensor(class_num_gender).to(device=device, dtype=torch.float)
        class_weight_age = torch.tensor(class_num_age).to(device=device, dtype=torch.float)

        # self.age_fc = nn.Linear(3,18)
        # init.xavier_uniform_(self.age_fc.weight.data)
        # self.gender_fc = nn.Linear(2,18)
        # init.xavier_uniform_(self.gender_fc.weight.data)
        # self.mask_fc = nn.Linear(3,18)
        # init.xavier_uniform_(self.mask_fc.weight.data)

        self.classifier_fc = nn.Linear(8,18)
        init.xavier_uniform_(self.classifier_fc.weight.data)

        self.criterion_age = nn.CrossEntropyLoss(weight=class_weight_age)
        self.criterion_gender = nn.CrossEntropyLoss(weight=class_weight_gender)
        self.criterion_mask = nn.CrossEntropyLoss(weight=class_weight_mask)
        self.criterion_final = nn.CrossEntropyLoss(weight=class_weight_final)

        self.lr = 1e-4
        self.optimizer_mask = optim.AdamW(self.model_mask.parameters(), lr= self.lr)
        self.optimizer_age = optim.AdamW(self.model_age.parameters(), lr= self.lr)
        self.optimizer_gender = optim.AdamW(self.model_gender.parameters(), lr= self.lr)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        output_mask = self.model_mask(x)
        output_gender = self.model_gender(x)
        output_age = self.model_age(x)

        # classify_mask = self.mask_fc(output_mask.detach())
        # classify_gender = self.gender_fc(output_gender.detach())
        # classify_age = self.age_fc(output_age.detach())

        # final_output = classify_mask + classify_gender + classify_age

        final_output = self.classifier_fc(torch.cat((output_mask.detach(), output_gender.detach(), output_age.detach()), dim=1))
        final_output = self.softmax(final_output)

        return final_output

    def get_loss(self, x, labels, flag):
        output_mask = self.model_mask(x)
        output_mask_loss = self.criterion_mask(output_mask, labels[0])
        output_gender = self.model_gender(x)
        output_gender_loss = self.criterion_gender(output_gender, labels[1])
        output_age = self.model_age(x)
        output_age_loss = self.criterion_age(output_age, labels[2])

        output_loss = [output_mask_loss, output_gender_loss, output_age_loss]



        if flag:
            output_mask_loss.backward(retain_graph=True)
            output_gender_loss.backward(retain_graph=True)
            output_age_loss.backward(retain_graph=True)

            self.optimizer_mask.step()
            self.optimizer_gender.step()
            self.optimizer_age.step()
        
        # classify_mask = self.mask_fc(output_mask.detach())
        # classify_gender = self.gender_fc(output_gender.detach())
        # classify_age = self.age_fc(output_age.detach())
        # final_output = classify_mask + classify_gender + classify_age
        
        final_output = self.classifier_fc(torch.cat((output_mask.detach(), output_gender.detach(), output_age.detach()), dim=1))
        final_output = self.softmax(final_output)
        final_output_loss = self.criterion_final(final_output, labels[3])
        _, preds_final = torch.max(final_output, 1)

        return output_loss, final_output_loss, preds_final








         

        

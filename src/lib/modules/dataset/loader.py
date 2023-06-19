from .ClassDataset import *
train_dataset = PandasDataset(expanded_train_df,resolve_paired_data,y="posture_group_x",
                            transform=train_transforms,target_transform=target_transform)
resolve_img = ResolveImage()
train_dataset = PandasDataset(train_df,resolve_img,y="posture_group",
                            transform=train_transforms,target_transform=target_transform)
test_dataset = PandasDataset(test_df,resolve_img,y="posture_group",
                            transform=test_transforms,target_transform=target_transform)

# %%
#len(train_dataset),len(test_dataset)

# %%
batch_size = 8

#if __name__ == "__main__":
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=0)#,pin_memory=True)# ,prefetch_factor=128)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,  num_workers=0)#,pin_memory=True)


#assert dataset size and index equals
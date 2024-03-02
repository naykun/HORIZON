from config import *
import lmdb

class BaseDataset(Dataset):
    def __init__(self, args, split='train'):
        self.args = args

        self.split = split
        self.img_root = os.path.join(BasicArgs.root_dir, 'dataset/pano/jpegs_pittsburgh_2021/')
        self.img_full_size = args.img_full_size

        self.clip_feats_env = lmdb.open(
                "/msrhyper-ddn/hai1/kun/streetlearn/clip_feat/jpegs_pittsburgh_2021_multiview/",
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        if split == 'train':
            # train data contain 85k
            with open(os.path.join(BasicArgs.root_dir, 'dataset/pano/pittsburgh_train.json'), 'r') as f:
                data = json.load(f)
                self.filelist = list(data.keys())
                self.caplist = list(data.values())

            self.transform = transforms.Compose([
                # transforms.RandomResizedCrop(self.args.img_full_size, scale=(0.85, 1.), ratio=(1., 1.)),
                transforms.Resize((self.args.img_full_size, self.args.img_full_size)),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        elif split == 'val':
            # test data contain 5k
            with open(os.path.join(BasicArgs.root_dir, 'dataset/pano/pittsburgh_val.json'), 'r') as f:
                data = json.load(f)
                self.filelist = list(data.keys())
                self.caplist = list(data.values())

            self.transform = transforms.Compose([
                transforms.Resize((self.args.img_full_size, self.args.img_full_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return self.args.max_train_samples
            else:
                return len(self.filelist)
        else:
            if getattr(self.args, 'max_eval_samples', None):
                return self.args.max_eval_samples
            else:
                return len(self.filelist)

    def __getitem__(self, i):
        try:
            img_name = self.filelist[i]
            caption = self.caplist[i]

            img_path = os.path.join(self.img_root, img_name)
            img = Image.open(img_path).convert('RGB')
            image = self.transform(img)
            image = repeat(image, 'c h w -> l c h w', l=self.args.max_video_len)  # [1,3,1024,1024]
            img_base_name = img_name.split(".")[0]
            hint_image_names = [os.path.join(self.img_root, f"{img_base_name}_{str(x)}") for x in range(4)]
            with self.clip_feats_env.begin(write=False) as txn:
                hint_img_feats = [pickle.loads(txn.get(image_id.encode()))["feature"] for image_id in hint_image_names]
            hint_img_feats = torch.stack(hint_img_feats)
            sample = {"label_imgs": image, 'input_text': caption, 'input_imghint_feats': hint_img_feats}
            return sample

        except Exception as e:
            logger.warning('Bad idx %s skipped because of %s' % (self.filelist[i], e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

from data import load_data
from model import PatchEmbedStem  , Transformer_encoder_layer , Transformer_Encoder , MiniTransformer , Classifier , VIT_classifier
from train_eval import train_and_eval


path = "/home/subhajit/Notebooks/Final/Coordinate_Graphs/cifar10_dataset"
batch_size  = 32
image_dim = 32
patch_size = 8
num_heads = 2
num_layers  = 2
embed_dim = 64
dropout  = 0.1
num_classes = 10
patch_dim = 3 * (patch_size * patch_size)
n_patches  = (image_dim //patch_size ) * (image_dim // patch_size)
device = "cuda:3"
train_loader_cifar , test_loader_cifar = load_data( path  , batch_size  = batch_size)

Patchembedding  = PatchEmbedStem(patch_dim = patch_dim,
                                embed_dim = embed_dim , 
                                patch_size  = patch_size
                                )
vit_transformer = MiniTransformer(n_patches = n_patches , 
                                 embed_dim = embed_dim ,
                                 num_heads = num_heads ,
                                 num_layers = num_layers,
                                 dropout  = dropout)


classifier  = Classifier(embed_dim = embed_dim , num_classes = num_classes)

model = VIT_classifier(Patchembedding , vit_transformer , classifier).to(device)

best_model, train_hist, test_hist = train_and_eval(model, train_loader_cifar, test_loader_cifar, device, optimizer=None, num_epochs=1000)


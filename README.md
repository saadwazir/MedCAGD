# MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image Segmentation - Accepted in ECCV 2026
> 🚧 **Work in Progress**
> ⏳ Source code and supporting files are being prepared.
> 📦 Public release coming soon.
> 🙏 Please wait for the official release. Thank you for your patience!
> 

Download Paper:
[https://arxiv.org/abs/2607.00409](https://arxiv.org/abs/2607.00409)

Please Cite it as following

```
@inproceedings{wazir2026medcagdcontextawaregateddecoder,
      title={MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image Segmentation}, 
      author={Saad Wazir, Patrick Dominique Vibild, Dinh Phu Tran, Seongah Kim, Daeyoung Kim},
      year={2026},
      eprint={2607.00409},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.00409}, 
}
```

![medcagd-poster](./medcagd-poster.jpg)

## Download Dataset from Huggingface.
Link: [https://huggingface.co/datasets/saadwazir/MedCAGD-Dataset-Collection](https://huggingface.co/datasets/saadwazir/MedCAGD-Dataset-Collection)

## Benchmark Results

<table>
<caption>Table: 1 - Comprehensive performance comparison across 9 medical image segmentation benchmarks. Average Dice scores are reported.</caption>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Params ↓</th>
<th rowspan="2">FLOPs ↓</th>
<th colspan="2">Skin</th>
<th colspan="2">Polyp</th>
<th colspan="2">Fundus</th>
<th colspan="2">Neoplasm</th>
<th>Cell</th>
<th>All</th>
</tr>
<tr>
<th>ISIC17</th><th>ISIC18</th>
<th>ETIS</th><th>ColonDB</th>
<th>DRIVE</th><th>FIVES</th>
<th>BUSI</th><th>ThyroidXL</th>
<th>CellSeg</th><th>Avg</th>
</tr>
</thead>
<tbody>
<tr><td>U-Net</td><td>34.53 M</td><td>65.53 G</td><td>83.07</td><td>86.67</td><td>76.85</td><td>83.95</td><td>71.20</td><td>75.77</td><td>74.04</td><td>71.16</td><td>71.52</td><td>77.14</td></tr>
<tr><td>AttnUNet</td><td>34.88 M</td><td>66.64 G</td><td>83.66</td><td>87.05</td><td>76.84</td><td>86.46</td><td>71.68</td><td>75.99</td><td>74.48</td><td>72.50</td><td>72.64</td><td>77.92</td></tr>
<tr><td>DeepLabv3+</td><td>39.76 M</td><td>14.92 G</td><td>83.84</td><td>88.64</td><td>90.73</td><td>91.92</td><td>69.59</td><td>75.12</td><td>76.81</td><td>73.46</td><td>71.90</td><td>80.22</td></tr>
<tr><td>UNet++</td><td><u>09.16 M</u></td><td>34.65 G</td><td>82.98</td><td>87.46</td><td>77.40</td><td>87.88</td><td>72.94</td><td><u>85.74</u></td><td>74.46</td><td>83.94</td><td>78.30</td><td>81.23</td></tr>
<tr><td>nnU-Net</td><td>31.29 M</td><td>55.26 G</td><td>83.23</td><td>88.53</td><td>80.13</td><td>91.63</td><td>75.43</td><td>76.10</td><td>76.46</td><td>86.08</td><td>83.53</td><td>82.34</td></tr>
<tr><td>PraNet</td><td>32.55 M</td><td>06.93 G</td><td>83.03</td><td>88.56</td><td>83.84</td><td>89.16</td><td>75.21</td><td>84.57</td><td>75.14</td><td>85.51</td><td>79.07</td><td>82.68</td></tr>
<tr><td>TransUNet</td><td>105.32 M</td><td>38.52 G</td><td>85.00</td><td>89.16</td><td>87.79</td><td>91.63</td><td>74.98</td><td>83.54</td><td>78.30</td><td>85.77</td><td>79.08</td><td>83.92</td></tr>
<tr><td>Swin-Unet</td><td>27.17 M</td><td>06.20 G</td><td>83.97</td><td>89.26</td><td>85.10</td><td>89.27</td><td>74.93</td><td>84.17</td><td>77.38</td><td>85.80</td><td>78.84</td><td>83.19</td></tr>
<tr><td>UCTransNet</td><td>65.60 M</td><td>56.70 G</td><td>83.27</td><td>89.18</td><td>87.35</td><td>91.65</td><td>75.42</td><td>84.74</td><td>79.53</td><td>85.82</td><td>79.33</td><td>84.03</td></tr>
<tr><td>UNeXt</td><td><strong>1.470 M</strong></td><td><strong>0.570 G</strong></td><td>82.74</td><td>87.78</td><td>74.03</td><td>83.84</td><td>74.77</td><td>76.60</td><td>74.71</td><td>84.46</td><td>75.71</td><td>79.40</td></tr>
<tr><td>VM-UNet</td><td>27.43 M</td><td><u>04.12 G</u></td><td><u>85.99</u></td><td>87.05</td><td>85.52</td><td>88.71</td><td>73.25</td><td>83.51</td><td>74.69</td><td>78.31</td><td>74.94</td><td>81.33</td></tr>
<tr><td>Swin-UMamba</td><td>60.00 M</td><td>68.00 G</td><td>83.40</td><td>87.62</td><td>86.63</td><td>87.97</td><td>73.32</td><td>82.66</td><td>73.38</td><td>84.96</td><td>75.56</td><td>81.72</td></tr>
<tr><td>EMCAD</td><td>26.76 M</td><td>05.60 G</td><td>85.95</td><td>90.96</td><td><u>92.29</u></td><td><u>92.31</u></td><td>77.15</td><td>82.51</td><td><u>80.25</u></td><td>83.33</td><td>79.13</td><td>84.87</td></tr>
<tr><td>MCADS</td><td>50.90 M</td><td>61.89 G</td><td>84.14</td><td><u>91.01</u></td><td>92.24</td><td>91.37</td><td><u>78.42</u></td><td>76.05</td><td>80.03</td><td><u>86.33</u></td><td><strong>86.68</strong></td><td><u>85.14</u></td></tr>
<tr><td><strong>Ours</strong></td><td>30.60 M</td><td>05.00 G</td><td><strong>86.61</strong></td><td><strong>91.56</strong></td><td><strong>93.47</strong></td><td><strong>93.27</strong></td><td><strong>81.63</strong></td><td><strong>87.50</strong></td><td><strong>83.47</strong></td><td><strong>88.02</strong></td><td><u>86.61</u></td><td><strong>88.01</strong></td></tr>
<tr><td>AutoSam*</td><td>41.56 M</td><td>25.11 G</td><td>-</td><td>-</td><td>79.70</td><td>83.00</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>Medical SAM3*</td><td>840.0 M</td><td>-</td><td>-</td><td>-</td><td>86.10</td><td>-</td><td>55.80</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</tbody></table>


<table>
<caption>Table: 2 - Performance comparison with SOTA methods on the Synapse multi-organ dataset.</caption>
<thead><tr><th>Method</th><th>Dice ↑</th><th>IoU ↑</th><th>HD95 ↓</th><th>Aorta</th><th>GB</th><th>KL</th><th>KR</th><th>Liver</th><th>PC</th><th>SP</th><th>SM</th></tr></thead><tbody><tr><td>U-Net</td><td>70.11</td><td>59.39</td><td>44.69</td><td>84.00</td><td>56.70</td><td>72.41</td><td>62.64</td><td>86.98</td><td>48.73</td><td>81.48</td><td>67.96</td></tr><tr><td>AttnUNet</td><td>71.70</td><td>68.09</td><td>26.01</td><td>84.04</td><td>66.42</td><td>57.26</td><td>84.53</td><td>81.28</td><td>73.87</td><td>66.06</td><td>60.17</td></tr><tr><td>UNet++</td><td>72.39</td><td>68.82</td><td>25.61</td><td>83.65</td><td>67.66</td><td>57.26</td><td>84.53</td><td>81.34</td><td>73.87</td><td>68.97</td><td>61.85</td></tr><tr><td>nnU-Net</td><td>75.33</td><td>71.47</td><td>19.34</td><td>77.06</td><td>73.27</td><td>76.34</td><td>84.53</td><td>79.98</td><td>73.34</td><td>77.62</td><td>60.52</td></tr><tr><td>PraNetV2</td><td>83.75</td><td>74.81</td><td>17.77</td><td>88.69</td><td>72.79</td><td>85.41</td><td>82.91</td><td><u>95.82</u></td><td>68.47</td><td><u>93.09</u></td><td><u>85.85</u></td></tr><tr><td>TransUNet</td><td>77.61</td><td>67.32</td><td>26.90</td><td>86.56</td><td>60.43</td><td>80.54</td><td>78.53</td><td>94.33</td><td>58.47</td><td>87.06</td><td>75.00</td></tr><tr><td>Swin-Unet</td><td>77.58</td><td>66.88</td><td>27.32</td><td>81.76</td><td>65.95</td><td>82.32</td><td>79.22</td><td>93.73</td><td>53.81</td><td>88.04</td><td>75.79</td></tr><tr><td>UCTransNet</td><td>79.08</td><td>75.41</td><td>15.59</td><td>83.06</td><td>81.35</td><td>77.24</td><td>78.23</td><td>85.76</td><td>74.77</td><td>81.89</td><td>70.31</td></tr><tr><td>UNETR*</td><td>78.35</td><td>-</td><td>18.59</td><td>89.80</td><td>56.30</td><td>85.60</td><td>84.52</td><td>94.57</td><td>60.47</td><td>85.00</td><td>70.46</td></tr><tr><td>MISSFormer*</td><td>81.96</td><td>-</td><td>18.20</td><td>86.99</td><td>68.65</td><td>85.21</td><td>82.00</td><td>94.41</td><td>65.67</td><td>91.92</td><td>80.81</td></tr><tr><td>U-Mamba</td><td>78.63</td><td>74.87</td><td>16.19</td><td>83.77</td><td>78.70</td><td>79.40</td><td>82.37</td><td>83.86</td><td>74.78</td><td>79.77</td><td>66.41</td></tr><tr><td>VM-UNet</td><td>73.39</td><td>71.61</td><td>27.97</td><td>63.57</td><td>72.62</td><td>77.98</td><td><strong>92.59</strong></td><td>79.44</td><td>70.80</td><td>55.58</td><td>74.55</td></tr><tr><td>EMCAD</td><td>83.63</td><td>74.65</td><td>15.68</td><td>88.14</td><td>68.87</td><td><u>88.08</u></td><td>84.10</td><td>95.26</td><td>68.51</td><td>92.17</td><td>83.92</td></tr><tr><td>MCADS</td><td>85.03</td><td><u>81.71</u></td><td><strong>11.11</strong></td><td>90.81</td><td><u>86.07</u></td><td>86.77</td><td>83.24</td><td>87.66</td><td><strong>83.55</strong></td><td>85.74</td><td>76.38</td></tr><tr><td><strong>Ours</strong></td><td><strong>87.00±0.2</strong></td><td><strong>83.77</strong></td><td><u>14.39</u></td><td><strong>92.28</strong></td><td><strong>90.31</strong></td><td><strong>89.72</strong></td><td><u>87.21</u></td><td>91.02</td><td><u>82.08</u></td><td>86.91</td><td>76.51</td></tr><tr><td>Self-Prompt SAM*</td><td><u>86.74</u></td><td>-</td><td>-</td><td><u>91.99</u></td><td>69.95</td><td>85.65</td><td>85.40</td><td><strong>97.39</strong></td><td>79.18</td><td><strong>94.38</strong></td><td><strong>89.94</strong></td></tr></tbody></table>

<table>
<caption>Table: 3 - Performance comparison with SOTA methods on the ACDC dataset.</caption>
<thead><tr><th>Method</th><th>Dice ↑</th><th>IoU ↑</th><th>HD95 ↓</th><th>RV</th><th>Myo</th><th>LV</th></tr></thead><tbody><tr><td>U-Net</td><td>81.56</td><td>73.41</td><td>6.9854</td><td>76.99</td><td>80.28</td><td>87.43</td></tr><tr><td>AttnUNet</td><td>82.37</td><td>73.94</td><td>6.1684</td><td>78.13</td><td>81.08</td><td>87.89</td></tr><tr><td>UNet++</td><td>81.97</td><td>73.92</td><td>6.4724</td><td>77.74</td><td>80.73</td><td>87.44</td></tr><tr><td>nnU-Net</td><td>82.66</td><td>74.27</td><td>6.1663</td><td>79.00</td><td>81.01</td><td>87.97</td></tr><tr><td>PraNetV2</td><td>83.74</td><td>76.13</td><td>6.3719</td><td>79.61</td><td>83.10</td><td>88.51</td></tr><tr><td>TransUNet</td><td>83.07</td><td>74.85</td><td>5.7578</td><td>79.16</td><td>81.65</td><td>88.41</td></tr><tr><td>Swin-Unet</td><td>82.61</td><td>74.59</td><td>6.1244</td><td>78.94</td><td>80.17</td><td>88.73</td></tr><tr><td>UCTransNet</td><td>84.89</td><td>77.57</td><td>5.6995</td><td>80.94</td><td>84.11</td><td><u>89.62</u></td></tr><tr><td>U-Mamba</td><td>84.18</td><td>76.47</td><td>5.8501</td><td>80.90</td><td>83.24</td><td>88.40</td></tr><tr><td>VM-UNet</td><td>81.02</td><td>72.74</td><td>7.0025</td><td>76.75</td><td>79.40</td><td>86.90</td></tr><tr><td>EMCAD</td><td><u>85.07</u></td><td><u>77.73</u></td><td><u>5.2472</u></td><td><u>81.58</u></td><td><u>84.23</u></td><td>89.42</td></tr><tr><td>MCADS</td><td>84.51</td><td>76.92</td><td>5.5595</td><td>81.16</td><td>83.27</td><td>89.09</td></tr><tr><td><strong>Ours</strong></td><td><strong>87.54±0.3</strong></td><td><strong>80.96</strong></td><td><strong>4.4057</strong></td><td><strong>85.27</strong></td><td><strong>86.23</strong></td><td><strong>91.11</strong></td></tr></tbody></table>

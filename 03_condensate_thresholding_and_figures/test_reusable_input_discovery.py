import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().with_name("# 3D TIFF channel thresholding.py")
SPEC = importlib.util.spec_from_file_location("threshold_pipeline", MODULE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class ReusableInputDiscoveryTests(unittest.TestCase):
    def test_arbitrary_tiff_folder_name_is_discovered(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            selected = root / "any_user_selected_name"
            selected.mkdir()
            (selected / "image.tif").touch()
            table = selected / pipeline.PARAMETER_TABLE_RELATIVE_PATH
            table.parent.mkdir(parents=True)
            table.write_text(
                "target,channel,threshold,min_object_size_voxels,max_object_size_voxels\n"
                "TargetA,C2,100,10,1000\n",
                encoding="utf-8",
            )
            self.assertEqual(pipeline.find_analysis_folders(root), [selected.resolve()])


if __name__ == "__main__":
    unittest.main()

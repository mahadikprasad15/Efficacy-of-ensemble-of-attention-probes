"""
Quick test script to verify Deception dataset integration.

Usage:
    python scripts/test_deception_dataset.py
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

from actprobe.datasets.loaders import DeceptionDataset

def test_dataset_loading():
    """Test basic dataset loading and formatting."""
    print("=" * 60)
    print("Testing Deception Dataset Loading")
    print("=" * 60)

    # Test loading each split
    for split in ["train", "validation", "test"]:
        print(f"\n📊 Loading {split} split...")
        try:
            ds = DeceptionDataset(split=split, limit=5)
            ds.load_data()

            print(f"   ✓ Loaded {len(ds)} scenarios")

            # Check first item
            if len(ds) > 0:
                item = ds[0]
                print(f"   ✓ Prompt preview: {item['prompt'][:100]}...")
                print(f"   ✓ Metadata keys: {list(item['metadata'].keys())}")
                print(f"   ✓ Has honest_completion: {'honest_completion' in item['metadata']}")
                print(f"   ✓ Has deceptive_completion: {'deceptive_completion' in item['metadata']}")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False

    return True

def test_dataset_content():
    """Test dataset content and structure."""
    print("\n" + "=" * 60)
    print("Testing Dataset Content")
    print("=" * 60)

    ds = DeceptionDataset(split="validation", limit=3)
    ds.load_data()

    print(f"\n📝 Examining {len(ds)} sample scenarios:\n")

    for i, item in enumerate(ds):
        print(f"--- Scenario {i+1} ---")
        print(f"ID: {item['metadata']['id']}")
        print(f"Prompt:\n{item['prompt'][:200]}...")
        print(f"\nHonest completion preview: {item['metadata']['honest_completion'][:100]}...")
        print(f"Deceptive completion preview: {item['metadata']['deceptive_completion'][:100]}...")
        print()

    return True

def test_split_sizes():
    """Verify split proportions are correct."""
    print("=" * 60)
    print("Testing Split Proportions")
    print("=" * 60)

    splits_info = {}

    for split in ["train", "validation", "test"]:
        ds = DeceptionDataset(split=split)
        ds.load_data()
        splits_info[split] = len(ds)

    total = sum(splits_info.values())

    print(f"\n📊 Dataset Split Information:")
    print(f"   Train:      {splits_info['train']:3d} scenarios ({splits_info['train']/total*100:.1f}%)")
    print(f"   Validation: {splits_info['validation']:3d} scenarios ({splits_info['validation']/total*100:.1f}%)")
    print(f"   Test:       {splits_info['test']:3d} scenarios ({splits_info['test']/total*100:.1f}%)")
    print(f"   Total:      {total:3d} scenarios")

    # Check proportions are roughly correct
    train_ratio = splits_info['train'] / total
    val_ratio = splits_info['validation'] / total
    test_ratio = splits_info['test'] / total

    expected_train = 0.7
    expected_val = 0.15
    expected_test = 0.15

    if abs(train_ratio - expected_train) < 0.05:
        print(f"   ✓ Train ratio ~{expected_train:.0%} ✓")
    else:
        print(f"   ⚠ Train ratio {train_ratio:.1%} differs from expected {expected_train:.0%}")

    if abs(val_ratio - expected_val) < 0.05:
        print(f"   ✓ Val ratio ~{expected_val:.0%} ✓")
    else:
        print(f"   ⚠ Val ratio {val_ratio:.1%} differs from expected {expected_val:.0%}")

    if abs(test_ratio - expected_test) < 0.05:
        print(f"   ✓ Test ratio ~{expected_test:.0%} ✓")
    else:
        print(f"   ⚠ Test ratio {test_ratio:.1%} differs from expected {expected_test:.0%}")

    return True

def main():
    """Run all tests."""
    print("\n" + "🔬 " * 20)
    print("DECEPTION DATASET INTEGRATION TEST")
    print("🔬 " * 20 + "\n")

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Dataset Content", test_dataset_content),
        ("Split Sizes", test_split_sizes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Deception dataset is ready to use.")
        print("\nNext steps:")
        print("1. Set CEREBRAS_API_KEY environment variable")
        print("2. Run: python scripts/cache_activations.py --dataset Deception --limit 10")
    else:
        print("⚠️  Some tests failed. Please check errors above.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

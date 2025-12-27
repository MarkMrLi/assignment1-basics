from tests.test_tokenizer import test_encode_iterable_memory_usage
import cProfile
import pstats


if __name__ == "__main__":
    # Create a profile object
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run the function to be profiled
    test_encode_iterable_memory_usage()
    
    # Stop profiling
    profiler.disable()
    
    # Create a stats object and sort by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Print the profiling results
    print("Profile results (sorted by cumulative time):")
    stats.print_stats(20)  # Show top 20 functions
    
    # Also print stats sorted by internal time
    print("\nProfile results (sorted by internal time):")
    stats.sort_stats('time')
    stats.print_stats(10)  # Show top 10 functions
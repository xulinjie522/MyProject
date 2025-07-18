//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++ (supporting code)
// |  |  |__   |  |  | | | |  version 3.12.0
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT

/*
This file implements a parser test suitable for fuzz testing. Given a byte
array data, it performs the following steps:

- j1 = from_bjdata(data)
- vec = to_bjdata(j1)
- j2 = from_bjdata(vec)
- assert(j1 == j2)
- vec2 = to_bjdata(j1, use_size = true, use_type = false)
- j3 = from_bjdata(vec2)
- assert(j1 == j3)
- vec3 = to_bjdata(j1, use_size = true, use_type = true)
- j4 = from_bjdata(vec3)
- assert(j1 == j4)

The provided function `LLVMFuzzerTestOneInput` can be used in different fuzzer
drivers.
*/

#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// see http://llvm.org/docs/LibFuzzer.html
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    try
    {
        // step 1: parse input
        std::vector<uint8_t> const vec1(data, data + size);
        json const j1 = json::from_bjdata(vec1);

        try
        {
            // step 2.1: round trip without adding size annotations to container types
            std::vector<uint8_t> const vec2 = json::to_bjdata(j1, false, false);

            // step 2.2: round trip with adding size annotations but without adding type annotations to container types
            std::vector<uint8_t> const vec3 = json::to_bjdata(j1, true, false);

            // step 2.3: round trip with adding size as well as type annotations to container types
            std::vector<uint8_t> const vec4 = json::to_bjdata(j1, true, true);

            // parse serialization
            json const j2 = json::from_bjdata(vec2);
            json const j3 = json::from_bjdata(vec3);
            json const j4 = json::from_bjdata(vec4);

            // serializations must match
            assert(json::to_bjdata(j2, false, false) == vec2);
            assert(json::to_bjdata(j3, true, false) == vec3);
            assert(json::to_bjdata(j4, true, true) == vec4);
        }
        catch (const json::parse_error&)
        {
            // parsing a BJData serialization must not fail
            assert(false);
        }
    }
    catch (const json::parse_error&)
    {
        // parse errors are ok, because input may be random bytes
    }
    catch (const json::type_error&)
    {
        // type errors can occur during parsing, too
    }
    catch (const json::out_of_range&)
    {
        // out of range errors may happen if provided sizes are excessive
    }

    // return 0 - non-zero return values are reserved for future use
    return 0;
}

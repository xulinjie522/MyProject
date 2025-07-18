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

- j1 = from_bson(data)
- vec = to_bson(j1)
- j2 = from_bson(vec)
- assert(j1 == j2)

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
        json const j1 = json::from_bson(vec1);

        if (j1.is_discarded())
        {
            return 0;
        }

        try
        {
            // step 2: round trip
            std::vector<uint8_t> const vec2 = json::to_bson(j1);

            // parse serialization
            json const j2 = json::from_bson(vec2);

            // serializations must match
            assert(json::to_bson(j2) == vec2);
        }
        catch (const json::parse_error&)
        {
            // parsing a BSON serialization must not fail
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
        // out of range errors can occur during parsing, too
    }

    // return 0 - non-zero return values are reserved for future use
    return 0;
}

// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// @notice Digest/context registry for a fixed two-client development activity.
/// Authentication of training bytes and model incorporation remain off-chain.
contract UploadRegistry {
    struct Binding {
        bytes32 activity;
        bytes32 client;
        uint64 round;
        bytes32 nonce;
        bytes32 baseModel;
        bytes32 layout;
        bytes32 codecSpec;
        bytes32 digest;
        address uploader;
        uint64 blockNumber;
    }
    mapping(bytes32 => address) public authorized;
    mapping(bytes32 => Binding) private records;
    mapping(bytes32 => bytes32) private nonceRecords;
    mapping(bytes32 => bytes32) private roundRecords;
    uint256 public recordCount;
    event Registered(bytes32 indexed record, bytes32 indexed activity, bytes32 indexed client,
                     uint64 round, bytes32 digest);

    constructor(bytes32[] memory clients, address[] memory accounts) {
        require(clients.length == accounts.length && clients.length > 0, "client setup");
        for (uint256 i; i < clients.length; ++i) {
            require(accounts[i] != address(0) && authorized[clients[i]] == address(0), "client identity");
            authorized[clients[i]] = accounts[i];
        }
    }

    function register(bytes32 record, Binding calldata b) external {
        require(msg.sender == authorized[b.client], "unauthorized client");
        require(record != bytes32(0) && b.activity != bytes32(0) && b.round > 0 &&
                b.nonce != bytes32(0) && b.digest != bytes32(0), "empty binding");
        require(b.uploader == address(0) && b.blockNumber == 0, "derived fields");
        Binding storage previous = records[record];
        if (previous.uploader != address(0)) {
            require(previous.activity == b.activity && previous.client == b.client &&
                    previous.round == b.round && previous.nonce == b.nonce &&
                    previous.baseModel == b.baseModel && previous.layout == b.layout &&
                    previous.codecSpec == b.codecSpec && previous.digest == b.digest,
                    "record collision");
            return;
        }
        bytes32 nonceKey = keccak256(abi.encode(b.activity, b.client, b.nonce));
        bytes32 roundKey = keccak256(abi.encode(b.activity, b.client, b.round));
        require(nonceRecords[nonceKey] == bytes32(0), "nonce collision");
        require(roundRecords[roundKey] == bytes32(0), "round collision");
        records[record] = b;
        records[record].uploader = msg.sender;
        records[record].blockNumber = uint64(block.number);
        nonceRecords[nonceKey] = record;
        roundRecords[roundKey] = record;
        ++recordCount;
        emit Registered(record, b.activity, b.client, b.round, b.digest);
    }

    function get(bytes32 record) external view returns (Binding memory) {
        return records[record];
    }
}
